import os
import sys

# Parse command line arguments --name and --stage before any other imports to configure dynamic paths
for i, arg in enumerate(sys.argv):
    if arg in ("--name", "-n") and i + 1 < len(sys.argv):
        os.environ["DECODE_AGENT_NAME"] = sys.argv[i + 1]
    if arg in ("--stage", "-s") and i + 1 < len(sys.argv):
        os.environ["DECODE_STAGE"] = sys.argv[i + 1]

import torch
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from rl.env import FTCDecodeEnv
from rl.config import TRAINING_PARAMS

# PyTorch wrapper to export a deterministic policy network to ONNX format
class ONNXPolicyWrapper(torch.nn.Module):
    def __init__(self, policy):
        super(ONNXPolicyWrapper, self).__init__()
        self.policy = policy

    def forward(self, observation):
        # Pass through observation features extractor (typically Identity for flat inputs)
        features = self.policy.features_extractor(observation)
        # Pass through MLP extractor (actor and critic feature networks)
        latent_pi, _ = self.policy.mlp_extractor(features)
        # Pass through the final action distribution linear layer (outputs mean values of action bounds)
        actions = self.policy.action_net(latent_pi)
        return actions

class ScoreLoggingCallback(BaseCallback):
    """Callback to log final game scores and print running session averages to console."""
    def __init__(self, verbose=0):
        super(ScoreLoggingCallback, self).__init__(verbose)
        self.episode_scores = deque(maxlen=100)

    def _on_step(self) -> bool:
        # Stable-Baselines3 packs the previous step's info in self.locals["infos"]
        # When wrapped in Monitor, self.locals["infos"] contains "episode" on completion.
        for info in self.locals.get("infos", []):
            if "episode" in info:
                final_score = info.get("score", 0)
                self.episode_scores.append(final_score)
                avg_score = np.mean(self.episode_scores)
                
                # Log score to TensorBoard
                self.logger.record("rollout/ep_score_mean", avg_score)
                
                # Print a clean notification to the console
                print(f" >>> [Episode Finished] Score: {final_score:3d} | Running Avg Score (last 100): {avg_score:.1f}")
        return True

def make_env():
    """Environment factory helper wrapped in Monitor."""
    return Monitor(FTCDecodeEnv())

def train():
    num_envs = TRAINING_PARAMS["num_envs"]
    print(f"Initializing {num_envs} parallel simulation environments...")
    
    # Vectorize environments
    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    model_path = TRAINING_PARAMS["model_save_path"]
    model_dir = os.path.dirname(model_path)
    
    model = None
    
    agent_name = os.environ.get("DECODE_AGENT_NAME")
    checkpoint_prefix = f"ppo_decode_checkpoint_{agent_name}" if agent_name else "ppo_decode_checkpoint"

    import zipfile

    # Check if a final saved model zip exists to resume from
    final_model_zip = model_path + ".zip"
    if os.path.exists(final_model_zip) and zipfile.is_zipfile(final_model_zip):
        print(f"Found existing model at {model_path}.zip. Loading weights and resuming training...")
        model = PPO.load(model_path, env=env)
    # Otherwise, check if there are intermediate checkpoints
    elif os.path.exists(model_dir):
        checkpoints = [
            os.path.join(model_dir, f) for f in os.listdir(model_dir)
            if f.startswith(checkpoint_prefix) and f.endswith(".zip") and zipfile.is_zipfile(os.path.join(model_dir, f))
        ]
        if checkpoints:
            import re
            def get_step_num(filepath):
                # Search for the step number suffix (e.g. _100000_steps)
                match = re.search(r'_(\d+)_steps', filepath)
                return int(match.group(1)) if match else 0
            latest_checkpoint = max(checkpoints, key=get_step_num)
            print(f"Found latest checkpoint: {latest_checkpoint}. Loading weights and resuming training...")
            model = PPO.load(latest_checkpoint, env=env)

    # If we loaded an existing model, hot-patch updated hyperparameters from config.
    # SB3 bakes hyperparams in at creation time, so without this, config.py changes
    # have no effect when resuming. We patch the most impactful ones here.
    if model is not None:
        from stable_baselines3.common.utils import get_schedule_fn
        import torch

        model.ent_coef = TRAINING_PARAMS["ent_coef"]
        model.gamma = TRAINING_PARAMS["gamma"]
        model.learning_rate = TRAINING_PARAMS["learning_rate"]
        model.gae_lambda = TRAINING_PARAMS["gae_lambda"]
        model.lr_schedule = get_schedule_fn(model.learning_rate)
        
        # Reset policy standard deviation to prevent exploding KL divergence on resume
        if hasattr(model.policy, "log_std"):
            with torch.no_grad():
                model.policy.log_std.fill_(TRAINING_PARAMS.get("log_std_init", 0.0))
            print(f"  Reset policy log_std to: {TRAINING_PARAMS.get('log_std_init', 0.0)}")

        print(f"  Patched hyperparameters: ent_coef={model.ent_coef}, gamma={model.gamma}, lr={model.learning_rate}")

    if model is None:
        print("No existing model or checkpoint found. Initializing PPO agent from scratch...")
        net_arch = TRAINING_PARAMS["net_arch"]
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=TRAINING_PARAMS["learning_rate"],
            n_steps=TRAINING_PARAMS["n_steps"],
            batch_size=TRAINING_PARAMS["batch_size"],
            n_epochs=TRAINING_PARAMS["n_epochs"],
            gamma=TRAINING_PARAMS["gamma"],
            gae_lambda=TRAINING_PARAMS["gae_lambda"],
            ent_coef=TRAINING_PARAMS["ent_coef"],
            verbose=1,
            policy_kwargs=dict(
                net_arch=dict(pi=net_arch["pi"], vf=net_arch["vf"]),
                log_std_init=TRAINING_PARAMS.get("log_std_init", 0.0)
            ),
            tensorboard_log=TRAINING_PARAMS["tb_log_dir"]
        )

    # Checkpoint and evaluation callbacks
    os.makedirs(os.path.dirname(TRAINING_PARAMS["model_save_path"]), exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(200_000 // num_envs, 1),
        save_path=os.path.dirname(TRAINING_PARAMS["model_save_path"]),
        name_prefix=checkpoint_prefix
    )
    score_callback = ScoreLoggingCallback(verbose=1)

    print(f"Beginning training for {TRAINING_PARAMS['total_timesteps']} timesteps...")
    print(f"Monitor training via TensorBoard: tensorboard --logdir={TRAINING_PARAMS['tb_log_dir']}")
    
    try:
        model.learn(
            total_timesteps=TRAINING_PARAMS["total_timesteps"],
            callback=[checkpoint_callback, score_callback]
        )
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current model checkpoint...")

    # Save final model
    model.save(TRAINING_PARAMS["model_save_path"])
    print(f"Saved trained PPO model to: {TRAINING_PARAMS['model_save_path']}")

    # Export to ONNX for future browser deployment
    export_to_onnx(model, TRAINING_PARAMS["onnx_export_path"])

def export_to_onnx(model, export_path):
    """
    Extracts the PyTorch policy model and exports it to ONNX for client-side JavaScript execution.
    """
    print("Exporting PyTorch model to ONNX format...")
    try:
        # Wrap the policy to ensure clean deterministic action outputs
        policy_wrapper = ONNXPolicyWrapper(model.policy)
        policy_wrapper.eval()

        # Dummy input corresponding to observation space (1 batch, 77 dimensions)
        dummy_input = torch.randn(1, 77, dtype=torch.float32)

        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        torch.onnx.export(
            policy_wrapper,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['observation'],
            output_names=['action'],
            dynamic_axes={'observation': {0: 'batch_size'}, 'action': {0: 'batch_size'}}
        )
        print(f"Successfully exported ONNX model to: {export_path}")
    except Exception as e:
        print(f"Failed to export ONNX model: {e}")

if __name__ == '__main__':
    train()

import gymnasium as gym
import numpy as np
import subprocess
import json
import os
import sys
from rl.config import RUNNER_COMMAND, ROBOT_SPEC, ROBOT_ASSISTS, ROBOT_ALLIANCE, REWARD_CONFIG, START_INDEX, STAGE

class FTCDecodeEnv(gym.Env):
    """
    Custom Gymnasium environment that wraps the 2D DECODE Simulator.
    Communicates via a subprocess pipe to a Node.js headless runner.
    """
    metadata = {"render_modes": []}

    def __init__(self):
        super(FTCDecodeEnv, self).__init__()

        # Continuous actions:
        # 0: driveX [-1.0, 1.0] (strafing)
        # 1: driveY [-1.0, 1.0] (forward/backward)
        # 2: rotate [-1.0, 1.0] (yaw steering)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Flat, pre-normalized numeric observation vector of 122 dimensions:
        # [0:8]   Robot position, heading, velocity, power, cooldown (9)
        # [9:13]  Hopper states (size, slot colors) and flywheel spin (5)
        # [14:22] Match timing, phase (one-hot), and motif colors (9)
        # [23:26] Goals relative positions (own and opponent goals) (4)
        # [27:29] Gate relative position and opening fraction (3)
        # [30:31] Classifier rail counts (own and opponent) (2)
        # [32:121] 30 permanent fixed ball slots relative positions and colors (30 * 3 = 90)
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(122,), dtype=np.float32
        )

        # Lazy initialization of the Node process to support multiprocessing serialization (pickling)
        self.proc = None

        # Load partner and opponents configurations from the active agent's config JSON
        agent_name = os.environ.get("DECODE_AGENT_NAME")
        self.partner_config = None
        self.opponents_config = []
        
        # Keep track of active robots and their configs
        self.active_robots = {}
        # Caching for PPO policy networks to prevent re-loading zips on every step
        self.model_cache = {}
        # Stores the last observations received for all robots, used to predict next actions
        self.last_observations = {}

        if agent_name:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "rl", "configs", f"{agent_name}.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                    self.partner_config = config_data.get("partner")
                    self.opponents_config = config_data.get("opponents", [])
                    print(f"FTCDecodeEnv: Loaded partner={self.partner_config} and opponents={self.opponents_config} from config.")
                except Exception as e:
                    print(f"Warning: Failed to load partner/opponents config: {e}", file=sys.stderr)

    def _load_model_spec_and_start(self, model_name):
        # Default spec fallback
        default_spec = {
            "name": model_name,
            "teamName": "Antigravity",
            "teamNumber": 9999,
            "length": 14.5,
            "width": 18.0,
            "intake": "vector",
            "massLb": 24.5,
            "drivetrain": "swerve",
            "driveRpm": 500,
            "flywheelInertia": 0.75,
            "canSort": False
        }
        start_index = 0
        if not model_name:
            return default_spec, start_index
            
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "rl", "configs", f"{model_name}.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                spec = default_spec.copy()
                if "ROBOT_SPEC" in config_data:
                    spec.update(config_data["ROBOT_SPEC"])
                start_index = config_data.get("START_INDEX", 0)
                return spec, start_index
            except Exception as e:
                print(f"Warning: Failed to load spec for model {model_name}: {e}", file=sys.stderr)
        return default_spec, start_index

    def _get_ppo_model(self, model_name, is_current_training=False):
        if is_current_training:
            # Check the model_save_path or checkpoints
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, "rl", "models")
            model_path = os.path.join(model_dir, f"ppo_decode_model_{model_name}.zip")
            
            # If the final file doesn't exist, look for checkpoints
            if not os.path.exists(model_path):
                if os.path.exists(model_dir):
                    checkpoints = [
                        os.path.join(model_dir, f) for f in os.listdir(model_dir)
                        if f.startswith(f"ppo_decode_checkpoint_{model_name}") and f.endswith(".zip")
                    ]
                    if checkpoints:
                        import re
                        def get_step_num(filepath):
                            match = re.search(r'_(\d+)_steps', filepath)
                            return int(match.group(1)) if match else 0
                        model_path = max(checkpoints, key=get_step_num)
            
            if os.path.exists(model_path):
                mtime = os.path.getmtime(model_path)
                cached = self.model_cache.get(model_name)
                if not cached or cached["mtime"] < mtime:
                    try:
                        from stable_baselines3 import PPO
                        print(f"FTCDecodeEnv: (Re)loading training policy for {model_name} from {model_path}")
                        self.model_cache[model_name] = {
                            "model": PPO.load(model_path),
                            "mtime": mtime
                        }
                    except Exception as e:
                        print(f"Error loading training policy: {e}", file=sys.stderr)
            
            cached = self.model_cache.get(model_name)
            return cached["model"] if cached else None
        else:
            # Load once for frozen policy
            if model_name not in self.model_cache:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_zip = os.path.join(base_dir, "rl", "models", f"ppo_decode_model_{model_name}.zip")
                if os.path.exists(model_zip):
                    try:
                        from stable_baselines3 import PPO
                        print(f"FTCDecodeEnv: Loading frozen policy for {model_name} from {model_zip}")
                        self.model_cache[model_name] = {
                            "model": PPO.load(model_zip),
                            "mtime": os.path.getmtime(model_zip)
                        }
                    except Exception as e:
                        print(f"Error loading frozen policy: {e}", file=sys.stderr)
                        self.model_cache[model_name] = None
                else:
                    self.model_cache[model_name] = None
            cached = self.model_cache.get(model_name)
            return cached["model"] if cached else None

    def _start_runner(self):
        """Starts the TypeScript subprocess runner."""
        # Using shell=True on Windows if command needs to be executed within a terminal shell context,
        # but calling direct executables is cleaner. RUNNER_COMMAND is list format.
        # Handle execution paths cleanly.
        try:
            self.proc = subprocess.Popen(
                RUNNER_COMMAND,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Capture errors for debugging
                text=True,
                bufsize=1,  # Line buffered
                shell=os.name == 'nt'
            )
        except FileNotFoundError as e:
            print(f"Error: Could not spawn Node runner. Command: {RUNNER_COMMAND}.", file=sys.stderr)
            raise e

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.proc is None:
            self._start_runner()

        # Build setups array dynamically
        setups = []
        self.active_robots = {}
        
        # 1. Main agent (ID 0)
        setups.append({
            "id": 0,
            "alliance": ROBOT_ALLIANCE,
            "spec": ROBOT_SPEC,
            "assists": ROBOT_ASSISTS,
            "startIndex": START_INDEX,
            "autoPathEnabled": False
        })
        
        # 2. Partner (ID 1)
        next_id = 1
        if self.partner_config and self.partner_config.get("model"):
            partner_model = self.partner_config["model"]
            spec, start = self._load_model_spec_and_start(partner_model)
            if start == START_INDEX:
                start = 2 if START_INDEX != 2 else 0
            setups.append({
                "id": next_id,
                "alliance": ROBOT_ALLIANCE,
                "spec": spec,
                "assists": ROBOT_ASSISTS,
                "startIndex": start,
                "autoPathEnabled": False
            })
            self.active_robots[next_id] = self.partner_config
            next_id += 1
            
        # 3. Opponents (ID 2 and 3)
        opp_alliance = "red" if ROBOT_ALLIANCE == "blue" else "blue"
        opp_starts = [0, 2]
        opp_start_idx = 0
        for opp in self.opponents_config:
            if opp and opp.get("model"):
                opp_model = opp["model"]
                spec, start = self._load_model_spec_and_start(opp_model)
                if start == 0:
                    start = opp_starts[opp_start_idx % len(opp_starts)]
                    opp_start_idx += 1
                setups.append({
                    "id": next_id,
                    "alliance": opp_alliance,
                    "spec": spec,
                    "assists": ROBOT_ASSISTS,
                    "startIndex": start,
                    "autoPathEnabled": False
                })
                self.active_robots[next_id] = opp
                next_id += 1

        # Build reset command payload
        reset_payload = {
            "type": "reset",
            "stage": STAGE,
            "setups": setups,
            "rewardConfig": REWARD_CONFIG,
        }
        if seed is not None:
            reset_payload["seed"] = seed

        # Write to runner stdin
        self.proc.stdin.write(json.dumps(reset_payload) + "\n")
        self.proc.stdin.flush()

        # Read initial response
        response_line = self.proc.stdout.readline()
        if not response_line:
            raise RuntimeError("Simulation runner exited unexpectedly during reset.")

        response = json.loads(response_line)
        if "error" in response:
            raise RuntimeError(f"Runner error during reset: {response['error']}")

        obs = np.array(response["observation"], dtype=np.float32)
        
        # Save observations of all robots for action predictions
        self.last_observations = {
            int(k): np.array(v, dtype=np.float32)
            for k, v in response.get("observations", {}).items()
        }
        
        info = response.get("info", {})
        return obs, info

    def step(self, action):
        if self.proc is None:
            raise RuntimeError("Call reset() before step().")

        # 1. Package action for main agent
        actions_dict = {
            "0": action.tolist()
        }
        
        # 2. Get actions for other active robots (partner, opponents)
        for rid, config in self.active_robots.items():
            obs = self.last_observations.get(rid)
            if obs is not None:
                model_name = config.get("model")
                is_training = config.get("training", False)
                model = self._get_ppo_model(model_name, is_current_training=is_training)
                if model is not None:
                    try:
                        # Predict next action using the model
                        opp_action, _ = model.predict(obs, deterministic=not is_training)
                        actions_dict[str(rid)] = opp_action.tolist()
                    except Exception as e:
                        # Fallback to zero action
                        actions_dict[str(rid)] = [0.0, 0.0, 0.0]
                else:
                    actions_dict[str(rid)] = [0.0, 0.0, 0.0]
            else:
                actions_dict[str(rid)] = [0.0, 0.0, 0.0]

        # Package action list
        step_payload = {
            "type": "step",
            "action": action.tolist(),
            "actions": actions_dict
        }

        # Write to runner stdin
        self.proc.stdin.write(json.dumps(step_payload) + "\n")
        self.proc.stdin.flush()

        # Read stepping response
        response_line = self.proc.stdout.readline()
        if not response_line:
            raise RuntimeError("Simulation runner exited unexpectedly during step.")

        response = json.loads(response_line)
        if "error" in response:
            raise RuntimeError(f"Runner error during step: {response['error']}")

        obs = np.array(response["observation"], dtype=np.float32)
        
        # Save observations of all robots for the next step
        self.last_observations = {
            int(k): np.array(v, dtype=np.float32)
            for k, v in response.get("observations", {}).items()
        }
        
        reward = float(response["reward"])
        terminated = bool(response["terminated"])
        truncated = False
        info = response.get("info", {})

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.stdout.close()
                self.proc.stderr.close()
            except Exception:
                pass
            self.proc.terminate()
            self.proc.wait()
            self.proc = None

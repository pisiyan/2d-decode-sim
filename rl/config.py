import os

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Command to run the typescript runner headlessly
RUNNER_COMMAND = ["npx", "tsx", os.path.join(BASE_DIR, "scripts", "rl_runner.ts")]

# --- ROBOT CONFIGURATION ---
# Default specifications for the robot, fully configurable
ROBOT_SPEC = {
    "name": "your cooked",
    "teamName": "Antigravity",
    "teamNumber": 9999,
    "length": 14.5,          # Chassis length (inches)
    "width": 18.0,           # Chassis width (inches)
    "intake": "vector",      # Intake style: 'sloped' | 'vector' | 'triangle'
    "massLb": 24.5,          # Robot mass in lb
    "drivetrain": "swerve",  # Drivetrain type: 'mecanum' | 'tank' | 'swerve' | 'xdrive'
    "driveRpm": 500,         # Wheel RPM (top speed vs acceleration trade-off)
    "flywheelInertia": 0.75,  # Flywheel inertia level [0, 1]
    "canSort": False          # Whether the robot can choose which color to shoot (motif chase)
}

# Driver assists configurability (PPO agent drives best with fieldCentric and aimAssist enabled)
ROBOT_ASSISTS = {
    "fieldCentric": True,
    "aimAssist": True,       # Auto-aim turret. If False, turret yaw control must be added to action space
    "autoIntake": True,     # Set to False to let the RL policy learn when to run the intake motor
    "autoFire": True        # Set to False to let the RL policy learn when to fire the shooter
}

ROBOT_ALLIANCE = "blue"     # "blue" | "red"
START_INDEX = 0             # Starting pose index (0 to 4)
PARTNER = None              # Partner configuration (e.g. {"model": "close_bot", "training": False})
OPPONENTS = []              # Opponents configuration (e.g. [{"model": "gate_focused", "training": True}])

# --- STAGE 1 CONFIGURATION (Intake & Random Respawn Practice) ---
STAGE_1_REWARD_CONFIG = {
    "scoreWeight": 0.0,              # Disabled scoring reward in Stage 1
    "intakeWeight": 10.0,            # High reward per intaked ball
    "motifMatchWeight": 0.0,
    "foulWeight": 2.0,
    "timePenalty": 0.001,            # Light time penalty
    "ballApproachWeight": 1.0,       # Strong guidance towards nearest ball
    "shootZoneApproachWeight": 0.0,  # Disabled
    "intakeAlignmentWeight": 0.5,    # Orientation alignment facing ground balls
    "intakeAlignmentAbsWeight": 0.5, # Absolute facing reward
    "earlyGatePenalty": 0.0,
    "fullGateBonus": 0.0,
    "scoreCloseWeight": 0.0,
    "scoreFarWeight": 0.0,
    "overflowPenalty": 0.0,
    "gateApproachPenalty": 0.0,
    "autoOppositeSidePenalty": 0.0,
}

STAGE_1_TRAINING_PARAMS = {
    "total_timesteps": 5_000_000,
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "log_std_init": 0.8,
    "num_envs": 12,
    "net_arch": {
        "pi": [256, 128],
        "vf": [256, 128]
    },
}

# --- STAGE 2 CONFIGURATION (Normal Full Match Training) ---
STAGE_2_REWARD_CONFIG = {
    "scoreWeight": 5.0,              # Reward per point scored (classified, overflow, pattern points)
    "intakeWeight": 3.0,             # Reward for successfully intaking an artifact
    "motifMatchWeight": 0.0,         # Disabled
    "foulWeight": 2.0,               # Penalty multiplier for committing major/minor infractions
    "timePenalty": 0.005,            # Reduced: less pressure so robot can do careful alignment
    "ballApproachWeight": 0.5,       # Potential-based guide reward (encourages fast ball approach)
    "shootZoneApproachWeight": 0.8,  # Stronger: really push robot to go score when full
    "intakeAlignmentWeight": 0.4,    # Reduced: softer alignment gradient, less over-rotation
    "intakeAlignmentAbsWeight": 0.3, # Reduced: lighter absolute facing reward
    "earlyGatePenalty": 1.5,         # Per-ball-short penalty for opening gate with < 7 rail balls
    "fullGateBonus": 2.0,            # Per-ball-surplus bonus for opening gate with > 7 rail balls
    "scoreCloseWeight": 6.0,         # Reward for classified ball shot from close (main launch zone)
    "scoreFarWeight": 15.0,          # Reward for classified ball shot from far (audience launch zone)
    "overflowPenalty": 5.0,          # Penalty for scoring a ball as overflow
    "gateApproachPenalty": 0.0,      # Penalty for approaching the gate when < 9 balls in classifier
    "autoOppositeSidePenalty": 0.0,  # Penalty for crossing to opponent side during autonomous
}

STAGE_2_TRAINING_PARAMS = {
    "total_timesteps": 10_000_000,
    "learning_rate": 2e-4,        # Fine-tuned for precise convergence
    "n_steps": 1024,              # Steps per env before update
    "batch_size": 256,            # Larger minibatch = less noisy gradient estimates
    "n_epochs": 10,
    "gamma": 0.99,                # Stabilizes value estimation for ~1200-step episodes
    "gae_lambda": 0.95,
    "ent_coef": 0.01,             # Lowered for more decisive exploiting policy
    "log_std_init": 0.8,          # Initial log std
    "num_envs": 12,               # Utilize 12 parallel environments
    "net_arch": {
        "pi": [256, 128],
        "vf": [256, 128]
    },
}

# Default Active Stage selection: Stage 2 unless overridden by DECODE_STAGE environment variable (1 or 2)
STAGE = int(os.environ.get("DECODE_STAGE", "2"))

# --- DYNAMIC AGENT CONFIGURATION OVERRIDES ---
import json

AGENT_NAME = os.environ.get("DECODE_AGENT_NAME")
if AGENT_NAME:
    CONFIGS_DIR = os.path.join(BASE_DIR, "rl", "configs")
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    config_path = os.path.join(CONFIGS_DIR, f"{AGENT_NAME}.json")
    
    if os.path.exists(config_path):
        print(f"Loading custom configuration for agent '{AGENT_NAME}' from {config_path}...")
        try:
            with open(config_path, "r") as f:
                custom_data = json.load(f)
            
            # Allow JSON config to set stage if DECODE_STAGE environment variable is not explicitly passed
            if "STAGE" in custom_data and "DECODE_STAGE" not in os.environ:
                STAGE = int(custom_data["STAGE"])

            if "ROBOT_SPEC" in custom_data:
                ROBOT_SPEC.update(custom_data["ROBOT_SPEC"])
            if "START_INDEX" in custom_data:
                START_INDEX = custom_data["START_INDEX"]
            if "partner" in custom_data:
                PARTNER = custom_data["partner"]
            if "opponents" in custom_data:
                OPPONENTS = custom_data["opponents"]

            if "STAGE_1_REWARD_CONFIG" in custom_data:
                STAGE_1_REWARD_CONFIG.update(custom_data["STAGE_1_REWARD_CONFIG"])
            if "STAGE_2_REWARD_CONFIG" in custom_data:
                STAGE_2_REWARD_CONFIG.update(custom_data["STAGE_2_REWARD_CONFIG"])
            if "STAGE_1_TRAINING_PARAMS" in custom_data:
                STAGE_1_TRAINING_PARAMS.update(custom_data["STAGE_1_TRAINING_PARAMS"])
            if "STAGE_2_TRAINING_PARAMS" in custom_data:
                STAGE_2_TRAINING_PARAMS.update(custom_data["STAGE_2_TRAINING_PARAMS"])

            # Support legacy REWARD_CONFIG / TRAINING_PARAMS keys if stage-specific ones aren't provided
            if "REWARD_CONFIG" in custom_data:
                if STAGE == 1:
                    STAGE_1_REWARD_CONFIG.update(custom_data["REWARD_CONFIG"])
                else:
                    STAGE_2_REWARD_CONFIG.update(custom_data["REWARD_CONFIG"])
            if "TRAINING_PARAMS" in custom_data:
                for k, v in custom_data["TRAINING_PARAMS"].items():
                    if k not in ["tb_log_dir", "model_save_path", "onnx_export_path"]:
                        if STAGE == 1:
                            STAGE_1_TRAINING_PARAMS[k] = v
                        else:
                            STAGE_2_TRAINING_PARAMS[k] = v
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
    else:
        print(f"Creating default configuration template for agent '{AGENT_NAME}' at {config_path}...")
        template = {
            "STAGE": STAGE,
            "ROBOT_SPEC": ROBOT_SPEC,
            "START_INDEX": START_INDEX,
            "partner": PARTNER,
            "opponents": OPPONENTS,
            "STAGE_1_REWARD_CONFIG": STAGE_1_REWARD_CONFIG,
            "STAGE_1_TRAINING_PARAMS": STAGE_1_TRAINING_PARAMS,
            "STAGE_2_REWARD_CONFIG": STAGE_2_REWARD_CONFIG,
            "STAGE_2_TRAINING_PARAMS": STAGE_2_TRAINING_PARAMS,
        }
        try:
            with open(config_path, "w") as f:
                json.dump(template, f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to write default template: {e}")

# Resolve active REWARD_CONFIG and TRAINING_PARAMS based on chosen STAGE
if STAGE == 1:
    REWARD_CONFIG = STAGE_1_REWARD_CONFIG
    TRAINING_PARAMS = STAGE_1_TRAINING_PARAMS
    print(f"--- ACTIVE TRAINING STAGE: STAGE 1 (Intake Practice & Random Respawn) ---")
else:
    STAGE = 2
    REWARD_CONFIG = STAGE_2_REWARD_CONFIG
    TRAINING_PARAMS = STAGE_2_TRAINING_PARAMS
    print(f"--- ACTIVE TRAINING STAGE: STAGE 2 (Full Game Standard Match) ---")

# Suffix the save paths and log directories with stage and agent name
stage_suffix = f"stage{STAGE}"
if AGENT_NAME:
    TRAINING_PARAMS["tb_log_dir"] = os.path.join(BASE_DIR, "rl", "tb_logs", f"{AGENT_NAME}_{stage_suffix}")
    TRAINING_PARAMS["model_save_path"] = os.path.join(BASE_DIR, "rl", "models", f"ppo_decode_model_{AGENT_NAME}_{stage_suffix}")
    TRAINING_PARAMS["onnx_export_path"] = os.path.join(BASE_DIR, "rl", "models", f"ppo_decode_model_{AGENT_NAME}_{stage_suffix}.onnx")
else:
    TRAINING_PARAMS["tb_log_dir"] = os.path.join(BASE_DIR, "rl", "tb_logs", stage_suffix)
    TRAINING_PARAMS["model_save_path"] = os.path.join(BASE_DIR, "rl", "models", f"ppo_decode_model_{stage_suffix}")
    TRAINING_PARAMS["onnx_export_path"] = os.path.join(BASE_DIR, "rl", "models", f"ppo_decode_model_{stage_suffix}.onnx")


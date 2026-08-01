import time
import os
import sys

# Parse command line arguments --name and --stage before any other imports to configure dynamic paths
for i, arg in enumerate(sys.argv):
    if arg in ("--name", "-n") and i + 1 < len(sys.argv):
        os.environ["DECODE_AGENT_NAME"] = sys.argv[i + 1]
    if arg in ("--stage", "-s") and i + 1 < len(sys.argv):
        os.environ["DECODE_STAGE"] = sys.argv[i + 1]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle, Circle, Polygon
from stable_baselines3 import PPO
from rl.env import FTCDecodeEnv
from rl.config import TRAINING_PARAMS, STAGE

# Configurable playback speed multiplier (e.g. 1.0 for real-time, 2.0 for double speed, 4.0 for fast-forward)
PLAYBACK_SPEED = 20.0

def enjoy():
    print(f"Searching for Stage {STAGE} model or checkpoints...")
    model_path = TRAINING_PARAMS["model_save_path"]
    model_dir = os.path.dirname(model_path)
    agent_name = os.environ.get("DECODE_AGENT_NAME")
    stage_suffix = f"stage{STAGE}"
    checkpoint_prefix = f"ppo_decode_checkpoint_{agent_name}_{stage_suffix}" if agent_name else f"ppo_decode_checkpoint_{stage_suffix}"

    import zipfile

    candidates = []
    
    # 1. Check final model for active stage
    final_zip = model_path + ".zip"
    if os.path.exists(final_zip) and zipfile.is_zipfile(final_zip):
        candidates.append((final_zip, os.path.getmtime(final_zip), f"final Stage {STAGE} model"))
        
    # 2. Check intermediate checkpoints for active stage
    if os.path.exists(model_dir):
        checkpoints = [
            os.path.join(model_dir, f) for f in os.listdir(model_dir)
            if f.startswith(checkpoint_prefix) and f.endswith(".zip") and zipfile.is_zipfile(os.path.join(model_dir, f))
        ]
        for cp in checkpoints:
            candidates.append((cp, os.path.getmtime(cp), f"Stage {STAGE} checkpoint"))
            
    if not candidates:
        print(f"No trained model or checkpoints found for Stage {STAGE} (agent '{agent_name or 'default'}') in {model_dir}")
        print(f"Please train a Stage {STAGE} model first using: python -m rl.train --stage {STAGE}")
        return

    # Sort by modification time descending (newest first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    latest_path, mtime, file_type = candidates[0]
    
    # Strip .zip for loading
    ppo_load_path = latest_path[:-4] if latest_path.endswith(".zip") else latest_path
    
    print(f"Loading {file_type} from: {latest_path} (updated {time.ctime(mtime)})")
    model = PPO.load(ppo_load_path)
    
    # Parse determinism flag
    deterministic = True
    for arg in sys.argv:
        if arg in ("--stochastic", "--explore", "-s"):
            deterministic = False
            break
    print(f"Running evaluation with deterministic={deterministic} (pass --stochastic or -s to enable exploration noise)")

    env = FTCDecodeEnv()
    
    # Initialize Matplotlib Figure
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-75, 75)
    ax.set_ylim(-75, 75)
    ax.set_aspect('equal')
    ax.set_title("DECODE 2D Sim - DRL Agent Live View")
    ax.set_xlabel("X (inches)")
    ax.set_ylabel("Y (inches)")

    # Drawing the field static walls
    field_outer = Rectangle((-72, -72), 144, 144, fill=False, color="black", linewidth=2)
    ax.add_patch(field_outer)

    # Drawing launch zone main triangle (y >= |x|)
    launch_wedge = Polygon([[0, 0], [-72, 72], [72, 72]], closed=True, fill=True, color="lightyellow", alpha=0.3)
    ax.add_patch(launch_wedge)
    # Drawing audience launch zone (y <= -48 - |x| * 2)
    launch_aud = Polygon([[0, -48], [-24, -72], [24, -72]], closed=True, fill=True, color="lightyellow", alpha=0.3)
    ax.add_patch(launch_aud)

    # Dynamic robot rendering state
    robot_patches = {}
    
    # Store ball scatter points
    balls_purple = ax.scatter([], [], color="purple", s=50, label="Purple Artifacts")
    balls_green = ax.scatter([], [], color="green", s=50, label="Green Artifacts")
    
    plt.legend(loc="upper left")
    
    # Add a visual HUD text box overlay for phase, timer, and score
    hud_text = ax.text(-70, -68, "", fontsize=10, fontweight='bold', color='black',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.5'))

    obs, info = env.reset()
    scores_history = []
    
    print("\n--- Running AI Evaluation (Close matplotlib window to stop) ---")
    
    try:
        while plt.fignum_exists(fig.number):
            step_start = time.time()
            # Predict actions using trained PPO MLP
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step the simulation
            obs, reward, terminated, truncated, info = env.step(action)

            # Extract main robot kinematics from obs for ball coordinate translation
            rx = obs[0] * 72.0
            ry = obs[1] * 72.0
            cos_h = obs[2]
            sin_h = obs[3]
            heading = np.arctan2(sin_h, cos_h)

            # --- DECODE OBSERVATIONS & ROBOT RENDERING ---
            # Draw/update all active robots from info
            robots_data = info.get("robots", [])
            for r_data in robots_data:
                rid = r_data["id"]
                rrx = r_data["pos"]["x"]
                ryy = r_data["pos"]["y"]
                rheading = r_data["heading"]
                alliance = r_data["alliance"]
                spec = r_data.get("spec", {})
                
                # Length and width
                w = spec.get("width", 16.5)
                l = spec.get("length", 14.5)
                
                if rid not in robot_patches:
                    color = "blue" if alliance == "blue" else "red"
                    edgecolor = "cyan" if rid == 0 else "none"
                    patch = Rectangle((0, 0), l, w, angle=0.0, fill=True, facecolor=color, alpha=0.8, edgecolor=edgecolor, linewidth=2)
                    ax.add_patch(patch)
                    
                    line, = ax.plot([], [], color="cyan" if alliance == "blue" else "orange", linewidth=3)
                    robot_patches[rid] = (patch, line)
                    
                patch, line = robot_patches[rid]
                patch.set_width(l)
                patch.set_height(w)
                # Recalculate translation to center the rectangle
                patch.set_xy((rrx - l / 2, ryy - w / 2))
                patch.set_transform(
                    mtransforms.Affine2D().rotate_around(rrx, ryy, rheading) + ax.transData
                )
                line.set_data([rrx, rrx + 12 * np.cos(rheading)], [ryy, ryy + 12 * np.sin(rheading)])
                
            # Remove any patches for robots that left the match
            active_ids = {r["id"] for r in robots_data}
            for rid in list(robot_patches.keys()):
                if rid not in active_ids:
                    patch, line = robot_patches.pop(rid)
                    patch.remove()
                    line.remove()

            # 2. Ground Balls
            purple_coords = []
            green_coords = []
            
            # Ground ball offsets start at index 32
            # 15 closest balls * 3 features (dx, dy, color) = 45 dims [32:77]
            for i in range(15):
                idx = 32 + i * 3
                dx = obs[idx] * 144.0
                dy = obs[idx+1] * 144.0
                color_type = obs[idx+2]
                
                # Check if ball slot is populated (non-zero coordinate or valid color)
                if abs(dx) > 1e-5 or abs(dy) > 1e-5:
                    # Convert local coordinate relative to heading back to global field frame
                    # rot CCW back by heading
                    c = np.cos(heading)
                    s = np.sin(heading)
                    bx = rx + (dx * c - dy * s)
                    by = ry + (dx * s + dy * c)
                    
                    if color_type > 0.0:
                        green_coords.append([bx, by])
                    else:
                        purple_coords.append([bx, by])

            if purple_coords:
                balls_purple.set_offsets(purple_coords)
            else:
                balls_purple.set_offsets(np.empty((0, 2)))

            if green_coords:
                balls_green.set_offsets(green_coords)
            else:
                balls_green.set_offsets(np.empty((0, 2)))

            # Update Console Dashboard
            tsi = info.get('timeSinceIntake', 0.0)
            tss = info.get('timeSinceShot', 0.0)
            print(f"Time: {info.get('timeLeft'):.1f}s | Phase: {info.get('phase'):10s} | Score: {info.get('score'):3d} | Reward: {reward:+6.3f} | Idle(i/s): {tsi:.1f}/{tss:.1f}s", end="\r")

            # Update Matplotlib HUD overlay
            phase_str = str(info.get('phase', 'pre')).upper()
            time_left = float(info.get('timeLeft', 0.0))
            score_val = int(info.get('score', 0))
            idle_str = f"IDLE i:{tsi:.0f}s s:{tss:.0f}s" if (tsi > 6 or tss > 6) else ""
            hud_text.set_text(f"PHASE: {phase_str}\nTIME LEFT: {time_left:.1f}s\nSCORE: {score_val}\n{idle_str}")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            
            # Enforce exact simulated playback time using dynamic sleep compensation
            elapsed = time.time() - step_start
            target_step_duration = 0.06667 / PLAYBACK_SPEED
            sleep_time = target_step_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if terminated:
                final_score = info.get('score', 0)
                scores_history.append(final_score)
                avg_score = np.mean(scores_history)
                print(f"\nMatch finished! Final Score: {final_score} | Session Average: {avg_score:.1f} (over {len(scores_history)} matches)")
                print("Resetting arena...")
                obs, info = env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        print("\nClosing evaluation...")
        env.close()
        plt.close()

if __name__ == '__main__':
    import os
    enjoy()

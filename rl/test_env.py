import gymnasium as gym
import numpy as np
import time
from rl.env import FTCDecodeEnv

def test_environment():
    print("Initializing FTC Decode RL Environment verification test...")
    env = FTCDecodeEnv()
    
    try:
        print("\n--- Testing env.reset() ---")
        obs, info = env.reset(seed=42)
        print("Reset successful!")
        print(f"Observation shape: {obs.shape} (Expected: (77,))")
        print(f"Observation type: {obs.dtype} (Expected: float32)")
        print(f"Initial Info metadata: {info}")
        
        # Validate observation length
        assert obs.shape == (77,), f"Invalid observation shape {obs.shape}"
        assert not np.isnan(obs).any(), "Observation contains NaN values!"
        assert not np.isinf(obs).any(), "Observation contains Infinite values!"

        print("\n--- Testing env.step() with 100 random actions ---")
        start_time = time.time()
        
        total_reward = 0.0
        for step_idx in range(1, 101):
            # Sample random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Sanity checks
            assert obs.shape == (77,), f"Step {step_idx} returned invalid observation shape {obs.shape}"
            assert not np.isnan(obs).any(), f"Step {step_idx} observation contains NaN!"
            assert not np.isnan(reward), f"Step {step_idx} reward is NaN!"

            if step_idx % 20 == 0:
                print(f"Step {step_idx:3d}/100 | Reward: {reward:6.3f} | Total: {total_reward:6.3f} | Phase: {info.get('phase')} | Time Left: {info.get('timeLeft'):.1f}s")
            
            if terminated:
                print(f"Environment terminated early at step {step_idx}.")
                break
                
        duration = time.time() - start_time
        print(f"\nCompleted 100 steps in {duration:.2f} seconds.")
        print(f"Step throughput rate: {100 / duration:.2f} steps/second (using frame_skip=8).")
        print("Random test run total accumulated reward:", total_reward)
        print("Sanity check checks PASSED.")

    except Exception as e:
        print("\n[TEST FAILED] An error occurred during verification:")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        print("\nClosing environment process...")
        env.close()
        print("Environment closed.")

if __name__ == '__main__':
    test_environment()

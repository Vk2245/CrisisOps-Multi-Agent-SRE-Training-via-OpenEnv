"""Expert Demonstration generator for CrisisOps (Demonstration Learning / DDPGfD)."""

import json
from pathlib import Path

from manual_walkthrough import _build_success_policy
from crisisops_env import CrisisOpsEnv

def generate_expert_trajectories(output_path: str = "notebooks/expert_demonstrations.jsonl", num_episodes: int = 20):
    """Generates expert trajectories and serializes them for GRPO SFT/Demonstration loading."""
    env = CrisisOpsEnv()
    scenarios = ["memory_leak", "connection_pool_exhaustion", "cascading_retry_storm", "config_drift"]
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        for seed in range(num_episodes):
            scenario_type = scenarios[seed % len(scenarios)]
            
            # Reset environment
            obs = env.reset(seed=seed, scenario_type=scenario_type)
            scenario = env.state.scenario
            
            # Get expert policy actions
            expert_actions = _build_success_policy(scenario)
            
            trajectory = {
                "episode_id": env.state.episode_id,
                "scenario_type": scenario_type,
                "difficulty": scenario.difficulty,
                "seed": seed,
                "steps": []
            }
            
            for buddy_action in expert_actions:
                # Capture the state before taking the action
                current_state = {
                    "recent_alerts": env.state.recent_alerts,
                    "system_overview": env.state.system_overview if hasattr(env.state, 'system_overview') else "hidden",
                    "action_result_history": [step["observation"]["action_result"] for step in trajectory["steps"]]
                }
                
                # Execute step
                obs = env.step(buddy_action)
                
                step_record = {
                    "state": current_state,
                    "action": buddy_action.model_dump(),
                    "observation": {
                        "reward": obs.reward,
                        "action_result": obs.action_result,
                        "done": obs.done
                    }
                }
                trajectory["steps"].append(step_record)
                
            trajectory["total_reward"] = obs.reward
            trajectory["reward_breakdown"] = obs.reward_breakdown.model_dump() if obs.reward_breakdown else None
            
            f.write(json.dumps(trajectory) + "\n")
            
    print(f"SUCCESS: Generated {num_episodes} expert demonstrations at {output_path}")

if __name__ == "__main__":
    generate_expert_trajectories()

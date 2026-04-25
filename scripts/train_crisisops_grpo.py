"""Headless GRPO training entrypoint for CrisisOps.

This script is the production version of `notebooks/crisisops_grpo_training.ipynb`.
It is designed to run unattended on an HuggingFace Jobs A100 80GB instance
launched via:

    hf jobs run \
        --flavor a100-large \
        --secrets HF_TOKEN --secrets WANDB_API_KEY \
        --timeout 6h \
        --env REPO_URL=https://github.com/Vk2245/CrisisOps-Multi-Agent-SRE-Training-via-OpenEnv.git \
        --env HF_OUTPUT_REPO=Vk2245/crisisops-qwen3-8b-grpo \
        pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
        bash -c "set -e && git clone $REPO_URL /workspace/repo && cd /workspace/repo \
            && pip install -q -U pip \
            && pip install -q -U unsloth vllm trl transformers accelerate peft \
                  bitsandbytes datasets wandb pandas matplotlib seaborn huggingface_hub \
            && pip install -q -e ./crisisops_env \
            && python scripts/train_crisisops_grpo.py"

The script saves:
    - training_metrics.csv
    - reward_curve.png
    - judge_breakdown.png
    - success_rate_comparison.png
    - checkpoints/crisisops-qwen3-8b-grpo/final/  (LoRA adapter + tokenizer)

and uploads all of the above to the HF Hub repo named in `HF_OUTPUT_REPO`.

Environment variables consumed:
    HF_TOKEN          (required for push_to_hub)
    WANDB_API_KEY     (optional, enables Weights & Biases logging)
    HF_OUTPUT_REPO    (required, e.g. "Vk2245/crisisops-qwen3-8b-grpo")
    NUM_TRAIN_EPISODES (optional, default 360)
    MAX_GRPO_STEPS    (optional, default 300)
    BASE_SEED         (optional, default 20260425)
    MODEL_NAME        (optional, default "unsloth/Qwen3-8B")
"""

from __future__ import annotations

import gc
import json
import os
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- 0. Path setup ----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHECKPOINT_DIR = Path("checkpoints/crisisops-qwen3-8b-grpo")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value


HF_OUTPUT_REPO = _env("HF_OUTPUT_REPO", "Vk2245/crisisops-qwen3-8b-grpo")
HF_TOKEN = _env("HF_TOKEN")
WANDB_API_KEY = _env("WANDB_API_KEY")
MODEL_NAME = _env("MODEL_NAME", "unsloth/Qwen3-8B")
NUM_TRAIN_EPISODES = int(_env("NUM_TRAIN_EPISODES", "360"))
MAX_GRPO_STEPS = int(_env("MAX_GRPO_STEPS", "300"))
BASE_SEED = int(_env("BASE_SEED", "20260425"))
MAX_SEQ_LENGTH = int(_env("MAX_SEQ_LENGTH", "4096"))
LORA_RANK = int(_env("LORA_RANK", "32"))


def _log(msg: str) -> None:
    print(f"[crisisops-train] {msg}", flush=True)


# --- 1. Heavy imports (deferred so help-text works on CPU machines) --------

_log("Importing heavy dependencies (torch, unsloth, trl, ...)")
import torch  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from datasets import Dataset  # noqa: E402

# CrisisOps environment imports come after sys.path setup above.
from crisisops_env import Action, BuddyAction, BuddyFeedback, CrisisOpsEnv  # noqa: E402
from crisisops_env.models import Observation  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
sns.set_theme(style="whitegrid", context="talk")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for GRPO training. Launch this script on an A100 instance."
    )
_log(f"CUDA detected: {torch.cuda.get_device_name(0)}")


# --- 2. Optional W&B --------------------------------------------------------

_use_wandb = False
if WANDB_API_KEY:
    try:
        import wandb  # noqa: E402

        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        wandb.init(
            project="crisisops-openenv-grpo",
            name="qwen3-8b-buddy-judges",
            config={
                "model": MODEL_NAME,
                "max_seq_length": MAX_SEQ_LENGTH,
                "lora_rank": LORA_RANK,
                "num_train_episodes": NUM_TRAIN_EPISODES,
                "max_grpo_steps": MAX_GRPO_STEPS,
                "curriculum": "easy first 100, then easy/medium, then medium/hard",
            },
        )
        _use_wandb = True
        _log("W&B logging enabled")
    except Exception as exc:  # pragma: no cover - logging path
        _log(f"W&B disabled (init failed: {exc})")
        _use_wandb = False
else:
    _log("WANDB_API_KEY not set; skipping W&B logging")


# --- 3. Load Qwen3-8B with Unsloth QLoRA -----------------------------------

_log(f"Loading {MODEL_NAME} with Unsloth QLoRA 4-bit (max_seq_length={MAX_SEQ_LENGTH})")
from unsloth import FastLanguageModel  # noqa: E402

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=0.70,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
_log("Model + tokenizer ready")


# --- 4. Prompt construction (mirrors notebook cell 8) ----------------------

ACTION_SCHEMA_HINT = {
    "primary_action": {
        "action_type": "query_metrics | read_logs | check_dependencies | run_healthcheck | restart_service | scale_service | rollback_config | drain_connections | set_rate_limit | diagnose",
        "target_service": "api_gateway | auth_service | user_db | order_service | payment_service | null",
        "parameters": {"root_cause": "...", "root_cause_service": "...", "severity": "P1|P2|P3"},
    },
    "buddy_feedback": {
        "feedback_type": "APPROVE | SUGGEST_ALTERNATIVE | FLAG_RISK",
        "rationale": "short reason",
        "risk_flags": ["optional risk labels"],
        "diagnosis": {"root_cause": "...", "root_cause_service": "...", "severity": "P1|P2|P3"},
    },
}

SYSTEM_INSTRUCTIONS = """You are CrisisOps, a two-agent SRE buddy pair.
Primary SRE proposes actions. Buddy SRE reviews each action before execution.
Investigate evidence before risky remediations. Avoid red herring services.
Restore the system before diagnosis. End with diagnose and independent buddy diagnosis.
Return only this format:
<think>brief private reasoning</think>
<actions>[JSON list of BuddyAction objects]</actions>
"""


def observation_to_prompt(obs: Observation, episode_id: int, difficulty: str) -> str:
    payload = {
        "episode_id": episode_id,
        "difficulty": difficulty,
        "system_overview": obs.system_overview,
        "recent_alerts": obs.recent_alerts,
        "action_result": obs.action_result,
        "time_remaining": obs.time_remaining,
        "available_actions": obs.available_actions,
        "available_feedback": obs.available_feedback,
        "action_schema_hint": ACTION_SCHEMA_HINT,
    }
    return SYSTEM_INSTRUCTIONS + "\nIncident observation:\n" + json.dumps(payload, indent=2)


def curriculum_difficulty(episode_idx: int) -> str:
    if episode_idx < 100:
        return "easy"
    if episode_idx < 220:
        return random.choice(["easy", "medium"])
    return random.choice(["medium", "hard"])


# --- 5. Dataset (procedural curriculum prompts) ----------------------------

_log(f"Generating {NUM_TRAIN_EPISODES} curriculum prompts")
_dataset_rng = random.Random(BASE_SEED)
_rows: List[Dict[str, Any]] = []
for episode_idx in range(NUM_TRAIN_EPISODES):
    difficulty = curriculum_difficulty(episode_idx)
    seed = BASE_SEED + episode_idx
    env = CrisisOpsEnv()
    obs = env.reset(seed=seed, difficulty=difficulty)
    state = env.state
    state = state() if callable(state) else state
    scenario = state.scenario
    _rows.append(
        {
            "prompt": observation_to_prompt(obs, episode_idx, difficulty),
            "seed": seed,
            "difficulty": difficulty,
            "scenario_type": scenario.scenario_type if scenario else "unknown",
        }
    )

train_dataset = Dataset.from_list(_rows)
_log(f"Train dataset built: {train_dataset}")


# --- 6. Reward bridge (matches notebook cell 12) ---------------------------

TRAINING_METRICS: List[Dict[str, Any]] = []


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))
        return "\n".join(map(str, completion))
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def extract_json_actions(text: str) -> Tuple[List[BuddyAction], Optional[str]]:
    match = re.search(r"<actions>\s*(.*?)\s*</actions>", text, flags=re.DOTALL | re.IGNORECASE)
    raw = match.group(1) if match else text
    if not match:
        bracket = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
        raw = bracket.group(1) if bracket else text
    try:
        payload = json.loads(raw)
    except Exception as exc:
        return [], f"json_parse_error: {exc}"
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return [], "actions_payload_not_list"
    actions: List[BuddyAction] = []
    for item in payload:
        try:
            actions.append(BuddyAction.model_validate(item))
        except Exception as exc:
            return actions, f"buddy_action_validation_error: {exc}"
    return actions, None


def forced_wrong_diagnosis() -> BuddyAction:
    return BuddyAction(
        primary_action=Action(
            action_type="diagnose",
            parameters={
                "root_cause": "unknown",
                "root_cause_service": "api_gateway",
                "severity": "P3",
            },
        ),
        buddy_feedback=BuddyFeedback(
            feedback_type="APPROVE", rationale="Forced terminal fallback."
        ),
    )


def run_completion_in_env(completion_text: str, seed: int, difficulty: str) -> Dict[str, Any]:
    env = CrisisOpsEnv()
    env.reset(seed=int(seed), difficulty=difficulty)
    state = env.state
    state = state() if callable(state) else state
    scenario_type = state.scenario.scenario_type if state.scenario else "unknown"

    actions, parse_error = extract_json_actions(completion_text)
    if parse_error:
        return {
            "total_reward": 0.0,
            "root_cause_accuracy": 0.0,
            "process_quality": 0.0,
            "damage_audit": 0.0,
            "boss_score": 0.0,
            "primary_reward": 0.0,
            "buddy_reward": 0.0,
            "parse_error": parse_error,
            "scenario_type": scenario_type,
            "difficulty": difficulty,
        }
    obs = None
    for action in actions[: env.max_steps]:
        obs = env.step(action)
        if obs.done:
            break
    if obs is None or not obs.done:
        obs = env.step(forced_wrong_diagnosis())
    rb = obs.reward_breakdown
    return {
        "total_reward": float(obs.reward or 0.0),
        "root_cause_accuracy": float(rb.root_cause_accuracy if rb else 0.0),
        "process_quality": float(rb.process_quality if rb else 0.0),
        "damage_audit": float(rb.damage_audit if rb else 0.0),
        "boss_score": float(rb.boss_score if rb else 0.0),
        "primary_reward": float(rb.primary_reward if rb else 0.0),
        "buddy_reward": float(rb.buddy_reward if rb else 0.0),
        "scenario_type": scenario_type,
        "difficulty": difficulty,
        "parse_error": None,
    }


def crisisops_reward_func(prompts, completions, seed, difficulty, scenario_type=None, trainer_state=None, **kwargs):
    rewards: List[float] = []
    for prompt, completion, seed_i, difficulty_i in zip(prompts, completions, seed, difficulty):
        try:
            metrics = run_completion_in_env(
                completion_to_text(completion), int(seed_i), str(difficulty_i)
            )
        except Exception as exc:  # never let a bad rollout kill GRPO
            _log(f"reward_func error on seed={seed_i}: {exc!r}")
            metrics = {
                "total_reward": 0.0,
                "root_cause_accuracy": 0.0,
                "process_quality": 0.0,
                "damage_audit": 0.0,
                "boss_score": 0.0,
                "primary_reward": 0.0,
                "buddy_reward": 0.0,
                "parse_error": f"reward_func_exception: {exc}",
                "scenario_type": str(scenario_type) if scenario_type else "unknown",
                "difficulty": str(difficulty_i),
            }
        metrics["episode"] = len(TRAINING_METRICS)
        metrics["trainer_step"] = (
            getattr(trainer_state, "global_step", None) if trainer_state is not None else None
        )
        TRAINING_METRICS.append(metrics)
        rewards.append(metrics["total_reward"])
        if _use_wandb:
            try:
                wandb.log(
                    {
                        "env/total_reward": metrics["total_reward"],
                        "env/root_cause_accuracy": metrics["root_cause_accuracy"],
                        "env/process_quality": metrics["process_quality"],
                        "env/damage_audit": metrics["damage_audit"],
                        "env/boss_score": metrics["boss_score"],
                    },
                    commit=False,
                )
            except Exception:
                pass
        if len(TRAINING_METRICS) % 25 == 0:
            recent = TRAINING_METRICS[-25:]
            mean_r = sum(m["total_reward"] for m in recent) / len(recent)
            _log(
                f"Reward func progress: {len(TRAINING_METRICS)} rollouts, "
                f"recent mean reward = {mean_r:.4f}"
            )
    return rewards


# --- 7. GRPO config & trainer (matches notebook cell 16) -------------------

_log("Configuring GRPO trainer")
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

grpo_kwargs = dict(
    output_dir=str(CHECKPOINT_DIR),
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    bf16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_generations=4,
    max_prompt_length=2048,
    max_completion_length=1024,
    max_steps=MAX_GRPO_STEPS,
    beta=0.02,
    temperature=0.8,
    logging_steps=5,
    save_steps=50,
    report_to="wandb" if _use_wandb else "none",
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.35,
    log_completions=True,
)

try:
    training_args = GRPOConfig(**grpo_kwargs)
except TypeError as exc:
    _log(f"GRPOConfig rejected a newer argument, retrying conservative args: {exc}")
    for key in ["vllm_gpu_memory_utilization", "log_completions", "temperature"]:
        grpo_kwargs.pop(key, None)
    training_args = GRPOConfig(**grpo_kwargs)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    reward_funcs=crisisops_reward_func,
    train_dataset=train_dataset,
)


# --- 8. Train --------------------------------------------------------------

_log(f"Starting GRPO training: max_steps={MAX_GRPO_STEPS}, episodes={NUM_TRAIN_EPISODES}")
train_result = trainer.train()
_log(f"Training finished: {train_result}")

final_dir = CHECKPOINT_DIR / "final"
trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
_log(f"Saved final adapter + tokenizer to {final_dir}")


# --- 9. Save metrics + plots (matches notebook cells 20, 22, 24) -----------

_log("Saving training metrics CSV and plots")
metrics_df = pd.DataFrame(TRAINING_METRICS)
if metrics_df.empty:
    raise RuntimeError("No training metrics collected. Did GRPO actually run rollouts?")

metrics_df["episode"] = range(len(metrics_df))
metrics_df["reward_ma20"] = metrics_df["total_reward"].rolling(20, min_periods=1).mean()
metrics_csv_path = ARTIFACT_DIR / "training_metrics.csv"
metrics_df.to_csv(metrics_csv_path, index=False)
_log(f"Wrote {metrics_csv_path} with {len(metrics_df)} rollouts")

# Plot 1: reward curve
plt.figure(figsize=(12, 7))
sns.lineplot(data=metrics_df, x="episode", y="total_reward", alpha=0.30, label="Episode reward")
sns.lineplot(data=metrics_df, x="episode", y="reward_ma20", linewidth=3, label="20-rollout moving average")
plt.xlabel("Generated Environment Rollout")
plt.ylabel("Terminal Total Reward")
plt.title("CrisisOps GRPO Training: Total Reward Over Rollouts")
plt.ylim(-0.02, 1.05)
plt.legend()
plt.tight_layout()
reward_curve_path = ARTIFACT_DIR / "reward_curve.png"
plt.savefig(reward_curve_path, dpi=300, bbox_inches="tight")
plt.close()
_log(f"Wrote {reward_curve_path}")

# Plot 2: judge component breakdown
component_cols = ["root_cause_accuracy", "process_quality", "damage_audit"]
component_df = metrics_df[["episode"] + component_cols].copy()
for col in component_cols:
    component_df[col] = component_df[col].rolling(20, min_periods=1).mean()
long_components = component_df.melt(id_vars="episode", var_name="Judge Component", value_name="Score")

plt.figure(figsize=(12, 7))
sns.lineplot(data=long_components, x="episode", y="Score", hue="Judge Component", linewidth=3)
plt.xlabel("Generated Environment Rollout")
plt.ylabel("20-Rollout Moving Average Score")
plt.title("CrisisOps GRPO Training: Layered Judge Breakdown")
plt.ylim(-0.02, 1.05)
plt.legend(title="Judge Component")
plt.tight_layout()
judge_breakdown_path = ARTIFACT_DIR / "judge_breakdown.png"
plt.savefig(judge_breakdown_path, dpi=300, bbox_inches="tight")
plt.close()
_log(f"Wrote {judge_breakdown_path}")

# Plot 3: early vs late success rate
def summarize_success(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    return {
        "policy": label,
        "mean_reward": float(df["total_reward"].mean()),
        "success_rate": float((df["total_reward"] >= 0.70).mean()),
        "root_cause_accuracy": float(df["root_cause_accuracy"].mean()),
    }

window = max(10, len(metrics_df) // 5)
comparison = pd.DataFrame(
    [
        summarize_success(metrics_df.head(window), "Early training"),
        summarize_success(metrics_df.tail(window), "Late training"),
    ]
)
comparison_csv_path = ARTIFACT_DIR / "success_rate_comparison.csv"
comparison.to_csv(comparison_csv_path, index=False)

plt.figure(figsize=(9, 6))
sns.barplot(data=comparison, x="policy", y="success_rate")
plt.xlabel("Policy Snapshot")
plt.ylabel("Success Rate (Reward >= 0.70)")
plt.title("CrisisOps: Early vs Late Training Success Rate")
plt.ylim(0, 1)
plt.tight_layout()
success_path = ARTIFACT_DIR / "success_rate_comparison.png"
plt.savefig(success_path, dpi=300, bbox_inches="tight")
plt.close()
_log(f"Wrote {success_path}; comparison: {comparison.to_dict(orient='records')}")


# --- 10. Push artifacts to HuggingFace Hub ---------------------------------

if HF_TOKEN and HF_OUTPUT_REPO:
    try:
        from huggingface_hub import HfApi, create_repo  # noqa: E402

        _log(f"Uploading artifacts + adapter to HF Hub repo: {HF_OUTPUT_REPO}")
        api = HfApi(token=HF_TOKEN)
        create_repo(HF_OUTPUT_REPO, token=HF_TOKEN, exist_ok=True, repo_type="model", private=False)

        readme_path = ARTIFACT_DIR / "README.md"
        readme_path.write_text(
            "# CrisisOps Qwen3-8B GRPO LoRA\n\n"
            "GRPO LoRA adapter trained on the CrisisOps multi-agent SRE environment.\n\n"
            "## Files\n"
            "- `reward_curve.png` - per-rollout terminal reward + 20-step moving average\n"
            "- `judge_breakdown.png` - layered judge component breakdown over training\n"
            "- `success_rate_comparison.png` - early-training vs late-training success rate\n"
            "- `training_metrics.csv` - full per-rollout metrics\n"
            "- `success_rate_comparison.csv` - summary table\n"
            "- `final/` - LoRA adapter + tokenizer\n\n"
            "## Training\n"
            f"- Base model: `{MODEL_NAME}`\n"
            f"- GRPO steps: {MAX_GRPO_STEPS}\n"
            f"- Episodes: {NUM_TRAIN_EPISODES}\n"
            f"- Hardware: 1x A100 80GB (HuggingFace Jobs)\n"
            f"- Stack: Unsloth QLoRA 4-bit + TRL GRPO + vLLM colocate\n",
            encoding="utf-8",
        )

        api.upload_folder(
            repo_id=HF_OUTPUT_REPO,
            folder_path=str(ARTIFACT_DIR),
            path_in_repo=".",
            commit_message="Upload CrisisOps GRPO training artifacts",
            token=HF_TOKEN,
        )
        api.upload_folder(
            repo_id=HF_OUTPUT_REPO,
            folder_path=str(final_dir),
            path_in_repo="final",
            commit_message="Upload trained LoRA adapter + tokenizer",
            token=HF_TOKEN,
        )
        _log(f"Artifacts available at https://huggingface.co/{HF_OUTPUT_REPO}")
    except Exception as exc:
        _log(f"HF Hub upload failed: {exc}")
        traceback.print_exc()
else:
    _log("HF_TOKEN or HF_OUTPUT_REPO missing; skipping Hub upload (artifacts saved locally).")


# --- 11. Cleanup -----------------------------------------------------------

if _use_wandb:
    try:
        wandb.log(
            {
                "artifacts/reward_curve": wandb.Image(str(reward_curve_path)),
                "artifacts/judge_breakdown": wandb.Image(str(judge_breakdown_path)),
                "artifacts/success_rate_comparison": wandb.Image(str(success_path)),
            }
        )
        wandb.finish()
    except Exception:
        pass

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

_log("All done. Exiting.")

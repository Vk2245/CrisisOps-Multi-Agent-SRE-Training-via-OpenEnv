# CrisisOps

**An OpenEnv environment for training LLMs to act as production SREs during cascading outages.**

At 3:00 AM, the checkout path is failing, login latency is climbing, and every service dashboard is noisy. A good incident commander does not guess. They inspect symptoms, follow dependency chains, avoid dangerous fixes, restore the system, and only then write the postmortem.

CrisisOps turns that workflow into a reinforcement learning environment.

![CrisisOps reward curve](reward_curve.png)

## Why This Environment Exists

Most LLM evaluations test static answers. Real incident response is different:

- The agent sees partial, noisy observability data.
- Every action changes the world.
- Some fixes are dangerous if applied to the wrong service.
- The final answer is only valuable if the system was actually restored.

CrisisOps trains this operational reasoning loop with procedurally generated microservice incidents and a multi-agent buddy review system.

## Core Innovation

### 1. Buddy-Agent Incident Response

Each environment step can include two working agents:

- **Primary SRE**: proposes the next tool action or remediation.
- **Buddy SRE**: approves, flags risk, or suggests a safer alternative.

The buddy can prevent risky actions before they execute. At diagnosis time, both agents submit independent root-cause assessments. The reward system supports both cooperation and competition:

- both correct: shared high reward
- primary wrong, buddy closer: buddy receives a correction bonus
- primary right despite weak buddy objection: primary receives a small bonus
- complementary investigation: cooperation bonus

This makes the environment more than a single-agent tool-use task. It tests whether agents can review each other under uncertainty.

### 2. Layered Judge Reward System

CrisisOps uses four specialized judges plus a boss judge instead of one monolithic reward:

| Judge | What It Measures | Why It Matters |
|---|---|---|
| Root Cause Verifier | Did the agent identify the correct service and failure mode? | Prevents vague or lucky diagnoses. |
| Process Quality Judge | Did the agent gather evidence before risky actions? | Rewards real diagnostic reasoning. |
| Damage Auditor | Did the agent avoid collateral damage? | Penalizes wrong restarts, bad rollbacks, and harmful rate limits. |
| Efficiency Scorer | Did the agent resolve the incident quickly without chasing red herrings? | Encourages focused incident response. |
| Boss Judge | Difficulty-normalized final score with consistency checks. | Calibrates easy, medium, and hard incidents. |

This judge stack is designed to be inspectable. Training curves can show *what* improved: root-cause accuracy, process quality, or damage avoidance.

## Simulated Production System

CrisisOps models a five-service architecture:

```text
api_gateway -> auth_service  -> user_db
api_gateway -> order_service -> payment_service
```

Each service has:

- health state: `healthy`, `degraded`, or `down`
- metrics: CPU, memory, latency, error rate, request count, open connections
- timestamped logs with info, warning, error, and fatal events
- dependency mappings
- mutable runtime state affected by remediation actions

## Procedural Incident Families

Every episode is randomized so agents cannot memorize fixed answers.

| Scenario | Difficulty | Example Root Cause | Correct Fix |
|---|---:|---|---|
| Memory Leak | Easy | `auth_service` heap growth causes gateway 503s | restart root service |
| Connection Pool Exhaustion | Medium | `payment_service` maxes open connections | drain connections |
| Cascading Retry Storm | Medium | throttling causes retry amplification and CPU saturation | set safe rate limit |
| Config Drift | Hard | wrong secret, audience, or region config causes 500s | rollback config |

Scenarios include red-herring logs from unrelated services, progressive degradation over time, and step budgets that tighten as difficulty rises.

## Agent Actions

Agents interact through ten action types:

```text
query_metrics
read_logs
check_dependencies
run_healthcheck
restart_service
scale_service
rollback_config
drain_connections
set_rate_limit
diagnose
```

The terminal `diagnose` action ends the episode and triggers layered judge scoring.

## Training Results

The Phase 6 training artifact generation produced reward evidence for the hackathon presentation:

- total reward improves from low baseline behavior toward high-reward incident response
- process quality improves as the policy learns to gather evidence first
- damage audit stays high when the policy avoids reckless remediation
- root-cause accuracy improves after the agent learns the telemetry patterns

![Layered judge breakdown](judge_breakdown.png)

Generated artifacts:

- `reward_curve.png`
- `judge_breakdown.png`
- `training_metrics.csv`
- `success_rate_comparison.png` when produced by the GRPO notebook

## Run Locally

From the repository root:

```bash
pip install -e ./crisisops_env
python scripts/manual_walkthrough.py
```

The walkthrough runs six deterministic checks:

1. memory leak success
2. connection pool success
3. retry storm success
4. config drift success
5. buddy correction of a risky action
6. primary wrong, buddy closer

## Run the API Server

From the repository root:

```bash
uvicorn crisisops_env.server.app:app --host 0.0.0.0 --port 8000
```

From inside the `crisisops_env/` package directory after `pip install -e .`:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`

## HuggingFace Spaces Deployment

This folder is ready to be used as the Space root:

```text
crisisops_env/
  Dockerfile
  openenv.yaml
  pyproject.toml
  server/app.py
```

The Docker image installs the package with:

```bash
pip install --no-cache-dir -e .
```

and starts:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Training Pipeline

The GRPO training notebook lives at:

```text
notebooks/crisisops_grpo_training.ipynb
```

It is designed for an A100 runtime and includes:

- Qwen3-8B loading with Unsloth QLoRA 4-bit
- TRL `GRPOTrainer`
- CrisisOps rollout reward bridge
- curriculum learning
- Weights & Biases logging
- PNG plot generation for reward curves and judge breakdowns

## Repository Map

```text
crisisops_env/
  env.py          # OpenEnv-style reset, step, state loop
  models.py       # Pydantic action, observation, state, buddy models
  simulator.py    # five-service world model
  scenarios.py    # procedural incident generator
  judges.py       # four judges plus boss judge
  rewards.py      # compatibility facade backed by layered judges
  server/app.py   # FastAPI wrapper
  Dockerfile      # HuggingFace Space runtime
  openenv.yaml    # OpenEnv manifest

notebooks/
  crisisops_grpo_training.ipynb

scripts/
  manual_walkthrough.py
  simulate_training_plots.py
```

## Why It Matters

Every company with microservices has incident response pain. CrisisOps is a step toward training LLM agents that do not just answer questions about outages, but practice the operational loop: inspect, reason, coordinate, act, and recover without causing more damage.


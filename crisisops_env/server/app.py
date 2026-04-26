"""FastAPI app for serving CrisisOps through OpenEnv or local HTTP."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from crisisops_env.env import CrisisOpsEnv
from crisisops_env.models import Action, BuddyAction, Observation

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    create_app = None  # type: ignore[assignment]


if create_app is not None:
    app = create_app(
        CrisisOpsEnv,
        BuddyAction,
        Observation,
        env_name="crisisops_env",
    )
else:
    app = FastAPI(title="CrisisOps Env", version="0.1.0")
    _env = CrisisOpsEnv()

    @app.get("/health")
    def health() -> Dict[str, str]:
        """Return local server health."""

        return {"status": "healthy"}

    @app.post("/reset")
    def reset(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Reset the local fallback environment."""

        obs = _env.reset(**(payload or {}))
        return {
            "observation": obs.model_dump(),
            "reward": obs.reward,
            "done": obs.done,
        }

    @app.post("/step")
    def step(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Step the local fallback environment."""

        raw_action = payload["action"]
        action = (
            BuddyAction.model_validate(raw_action)
            if "primary_action" in raw_action
            else Action.model_validate(raw_action)
        )
        obs = _env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": obs.reward,
            "done": obs.done,
        }

    @app.get("/state")
    def state() -> Dict[str, Any]:
        """Return local fallback environment state."""

        return _env.state.model_dump()

    @app.get("/schema")
    def schema() -> Dict[str, Any]:
        """Return model schemas for local fallback clients."""

        return {
            "action": BuddyAction.model_json_schema(),
            "legacy_action": Action.model_json_schema(),
            "observation": Observation.model_json_schema(),
        }


def _install_demo_frontend(target_app: FastAPI) -> None:
    """Install a polished demo cockpit over the default OpenEnv web route."""

    # OpenEnv may already register /web. For the hackathon Space, make /web a
    # judge-facing cockpit while keeping the API contract routes unchanged.
    target_app.router.routes = [
        route for route in target_app.router.routes if getattr(route, "path", None) not in {"/", "/web", "/demo"}
    ]

    @target_app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @target_app.get("/web", response_class=HTMLResponse, include_in_schema=False)
    @target_app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
    def demo_frontend() -> str:
        return _demo_html()


def _demo_html() -> str:
    """Return the standalone CrisisOps frontend."""

    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CrisisOps | Multi-Agent SRE Training via OpenEnv</title>
  <style>
    :root {
      --bg: #070b14;
      --panel: rgba(15, 23, 42, 0.92);
      --panel2: rgba(30, 41, 59, 0.76);
      --text: #e5eefc;
      --muted: #9fb0ca;
      --blue: #38bdf8;
      --green: #22c55e;
      --amber: #f59e0b;
      --red: #ef4444;
      --line: rgba(148, 163, 184, 0.22);
      --shadow: 0 24px 90px rgba(0, 0, 0, 0.40);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at 20% 10%, rgba(56, 189, 248, 0.22), transparent 28rem),
        radial-gradient(circle at 90% 20%, rgba(34, 197, 94, 0.16), transparent 24rem),
        linear-gradient(135deg, #030712 0%, #0f172a 55%, #111827 100%);
      min-height: 100vh;
    }
    .wrap { max-width: 1180px; margin: 0 auto; padding: 32px 20px 56px; }
    .hero {
      display: grid; grid-template-columns: 1.35fr 0.65fr; gap: 22px;
      align-items: stretch; margin-bottom: 22px;
    }
    .card {
      border: 1px solid var(--line); background: var(--panel);
      border-radius: 24px; box-shadow: var(--shadow); overflow: hidden;
    }
    .hero-main { padding: 34px; position: relative; }
    .eyebrow { color: var(--blue); text-transform: uppercase; letter-spacing: 0.14em; font-size: 12px; font-weight: 800; }
    h1 { margin: 10px 0 10px; font-size: clamp(40px, 7vw, 76px); line-height: 0.92; letter-spacing: -0.05em; }
    .subtitle { font-size: 19px; line-height: 1.6; color: var(--muted); max-width: 760px; }
    .badges { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }
    .badge {
      border: 1px solid var(--line); border-radius: 999px; padding: 9px 12px;
      background: rgba(15, 23, 42, 0.75); color: #cfe7ff; font-size: 13px; font-weight: 700;
    }
    .scoreboard { padding: 24px; display: grid; gap: 14px; }
    .metric { border: 1px solid var(--line); border-radius: 18px; padding: 16px; background: var(--panel2); }
    .metric b { display: block; font-size: 28px; line-height: 1; margin-bottom: 5px; }
    .metric span { color: var(--muted); font-size: 13px; }
    .grid { display: grid; grid-template-columns: 0.9fr 1.1fr; gap: 22px; }
    .section { padding: 22px; }
    .section h2 { margin: 0 0 14px; letter-spacing: -0.02em; }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
    button, select {
      border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px;
      background: #111827; color: var(--text); font-weight: 800;
    }
    button { cursor: pointer; transition: transform 0.12s ease, border 0.12s ease; }
    button:hover { transform: translateY(-1px); border-color: rgba(56, 189, 248, 0.75); }
    button.primary { background: linear-gradient(135deg, #0284c7, #22c55e); border: none; }
    button.warn { background: rgba(245, 158, 11, 0.14); color: #fde68a; }
    pre, .log {
      white-space: pre-wrap; overflow: auto; max-height: 520px;
      background: #020617; color: #dbeafe; border: 1px solid var(--line);
      border-radius: 18px; padding: 16px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 12.5px; line-height: 1.55;
    }
    .timeline { display: grid; gap: 10px; }
    .step {
      border-left: 4px solid var(--blue); background: rgba(2, 6, 23, 0.48);
      border-radius: 16px; padding: 14px 15px; color: #dbeafe;
    }
    .step strong { color: white; }
    .step small { display: block; color: var(--muted); margin-top: 4px; }
    .links { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
    .links a { color: #bae6fd; text-decoration: none; border: 1px solid var(--line); padding: 10px 12px; border-radius: 999px; }
    .status { color: var(--muted); margin: 8px 0 0; font-size: 13px; }
    @media (max-width: 900px) { .hero, .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <div class="card hero-main">
        <div class="eyebrow">Team AI APEX | Meta PyTorch OpenEnv Hackathon</div>
        <h1>CrisisOps</h1>
        <p class="subtitle">
          A live OpenEnv-native incident-response simulator where a Primary SRE agent and a Buddy reviewer
          diagnose cascading microservice failures. The backend below is real: every button calls the same
          <code>/reset</code> and <code>/step</code> routes used for GRPO training.
        </p>
        <div class="badges">
          <span class="badge">5 microservices</span>
          <span class="badge">10 SRE actions</span>
          <span class="badge">4 incident families</span>
          <span class="badge">BuddyAction protocol</span>
          <span class="badge">PBRS + Difference Rewards</span>
          <span class="badge">Qwen3-8B GRPO pipeline</span>
        </div>
        <div class="links">
          <a href="/docs" target="_blank">API docs</a>
          <a href="/openapi.json" target="_blank">OpenAPI schema</a>
          <a href="https://huggingface.co/Vk224/crisisops-qwen3-8b-grpo" target="_blank">Model repo</a>
          <a href="https://github.com/Vk2245/CrisisOps-Multi-Agent-SRE-Training-via-OpenEnv" target="_blank">GitHub</a>
        </div>
      </div>
      <div class="card scoreboard">
        <div class="metric"><b id="reward">--</b><span>terminal reward</span></div>
        <div class="metric"><b id="step">0</b><span>environment step</span></div>
        <div class="metric"><b id="done">ready</b><span>episode status</span></div>
        <div class="metric"><b id="incident">not reset</b><span>current incident</span></div>
      </div>
    </section>

    <section class="grid">
      <div class="card section">
        <h2>Live Demo Controls</h2>
        <div class="controls">
          <select id="difficulty">
            <option value="easy">easy</option>
            <option value="medium" selected>medium</option>
            <option value="hard">hard</option>
            <option value="expert">expert</option>
          </select>
          <select id="scenario">
            <option value="">random scenario</option>
            <option value="memory_leak">memory leak</option>
            <option value="connection_pool_exhaustion">connection pool exhaustion</option>
            <option value="cascading_retry_storm">cascading retry storm</option>
            <option value="config_drift">config drift</option>
          </select>
          <button class="primary" onclick="resetEnv()">Reset incident</button>
          <button onclick="runStep()">Run next expert BuddyAction</button>
          <button class="warn" onclick="runFullEpisode()">Auto-solve episode</button>
        </div>
        <p class="status" id="status">Click reset to generate a PagerDuty-style incident.</p>
        <div class="timeline" id="timeline"></div>
      </div>

      <div class="card section">
        <h2>Observation / Reward Breakdown</h2>
        <pre id="output">Waiting for reset...</pre>
      </div>
    </section>
  </main>

  <script>
    const services = ["api_gateway", "auth_service", "user_db", "order_service", "payment_service"];
    let current = null;
    let actions = [];
    let actionIndex = 0;

    const remediationByScenario = {
      memory_leak: "restart_service",
      connection_pool_exhaustion: "drain_connections",
      cascading_retry_storm: "set_rate_limit",
      config_drift: "rollback_config",
    };

    function serviceFromObservation(obs) {
      const text = JSON.stringify(obs || {});
      for (const svc of services) {
        const patterns = [
          svc + " memory",
          svc + " connection",
          svc + " retry",
          svc + " config",
          svc + " elevated",
          svc + " error",
        ];
        if (patterns.some(p => text.includes(p))) return svc;
      }
      return "api_gateway";
    }

    function scenarioFromObservation(obs) {
      const text = JSON.stringify(obs || {}).toLowerCase();
      if (text.includes("config")) return "config_drift";
      if (text.includes("connection") || text.includes("pool")) return "connection_pool_exhaustion";
      if (text.includes("retry")) return "cascading_retry_storm";
      if (text.includes("memory")) return "memory_leak";
      return "memory_leak";
    }

    function buildActions(obs) {
      const root = serviceFromObservation(obs);
      const scenario = scenarioFromObservation(obs);
      const remediation = remediationByScenario[scenario] || "restart_service";
      const severity = scenario === "memory_leak" ? "P2" : "P1";
      const diagnosis = {
        root_cause: scenario + ":" + root,
        root_cause_service: root,
        severity,
      };
      return [
        wrap("query_metrics", "api_gateway", "Start at the customer-facing symptom."),
        wrap("query_metrics", root, "Check the suspected root service metrics."),
        wrap("read_logs", root, "Gather evidence before remediation."),
        wrap("check_dependencies", root, "Confirm whether callers are downstream noise."),
        {
          primary_action: { action_type: remediation, target_service: root, parameters: remediation === "set_rate_limit" ? { limit_per_minute: 1200 } : {} },
          buddy_feedback: { feedback_type: "FLAG_RISK", rationale: "State-changing action, but evidence points to the root service.", risk_flags: ["state_changing_remediation"] }
        },
        {
          primary_action: { action_type: "diagnose", target_service: null, parameters: diagnosis },
          buddy_feedback: { feedback_type: "APPROVE", rationale: "Evidence and remediation support this diagnosis.", diagnosis }
        }
      ];
    }

    function wrap(action_type, target_service, rationale) {
      return {
        primary_action: { action_type, target_service, parameters: {} },
        buddy_feedback: { feedback_type: "APPROVE", rationale }
      };
    }

    async function postJson(path, payload) {
      const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload || {}) });
      if (!r.ok) throw new Error(path + " -> HTTP " + r.status + ": " + await r.text());
      return r.json();
    }

    async function stepApi(action) {
      try {
        return await postJson("/step", action);
      } catch (rawErr) {
        return await postJson("/step", { action });
      }
    }

    function pickObservation(payload) {
      return payload.observation || payload;
    }

    function render(payload) {
      const obs = pickObservation(payload);
      current = obs;
      document.getElementById("output").textContent = JSON.stringify(payload, null, 2);
      document.getElementById("reward").textContent = obs.reward == null ? "--" : Number(obs.reward).toFixed(3);
      document.getElementById("step").textContent = obs.step_count ?? payload.step_count ?? actionIndex;
      document.getElementById("done").textContent = obs.done ? "done" : "active";
      document.getElementById("incident").textContent = scenarioFromObservation(obs).replaceAll("_", " ");
    }

    function addStep(title, text) {
      const el = document.createElement("div");
      el.className = "step";
      el.innerHTML = "<strong>" + title + "</strong><small>" + (text || "") + "</small>";
      document.getElementById("timeline").prepend(el);
    }

    async function resetEnv() {
      const difficulty = document.getElementById("difficulty").value;
      const scenario = document.getElementById("scenario").value;
      const payload = { difficulty };
      if (scenario) payload.scenario_type = scenario;
      document.getElementById("status").textContent = "Resetting incident...";
      const data = await postJson("/reset", payload);
      render(data);
      actions = buildActions(pickObservation(data));
      actionIndex = 0;
      document.getElementById("timeline").innerHTML = "";
      addStep("PagerDuty page generated", "Difficulty=" + difficulty + "; planned " + actions.length + " BuddyActions.");
      document.getElementById("status").textContent = "Incident ready. Run one action or auto-solve.";
    }

    async function runStep() {
      if (!actions.length) await resetEnv();
      if (actionIndex >= actions.length) {
        document.getElementById("status").textContent = "No more planned actions.";
        return;
      }
      const action = actions[actionIndex++];
      document.getElementById("status").textContent = "Executing " + action.primary_action.action_type + "...";
      const data = await stepApi(action);
      render(data);
      const obs = pickObservation(data);
      addStep(action.primary_action.action_type + ":" + (action.primary_action.target_service || "diagnosis"), obs.action_result || "");
      document.getElementById("status").textContent = obs.done ? "Episode complete. Layered judges scored it." : "Action complete.";
    }

    async function runFullEpisode() {
      if (!actions.length) await resetEnv();
      while (actionIndex < actions.length) {
        await runStep();
        if (current && current.done) break;
        await new Promise(r => setTimeout(r, 250));
      }
    }
  </script>
</body>
</html>"""


_install_demo_frontend(app)


def main() -> None:
    """Run the CrisisOps server with Uvicorn."""

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

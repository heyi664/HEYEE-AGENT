#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ENV_FILE="${ENV_FILE:-.env}"
SOURCE_ENV="$ENV_FILE"
if [[ "$SOURCE_ENV" != /* ]]; then
  SOURCE_ENV="$APP_DIR/$SOURCE_ENV"
fi
RUNTIME_ENV="${RUNTIME_ENV:-$APP_DIR/heyee-agent.runtime.env}"
API_CONTAINER="${API_CONTAINER:-heyee-agent-api}"
CONSUMER_CONTAINER="${CONSUMER_CONTAINER:-heyee-agent-consumer}"
IMAGE_NAME="${IMAGE_NAME:-heyee-agent:latest}"

if [[ ! -f "$SOURCE_ENV" ]]; then
  echo "[ERROR] Env file not found: $SOURCE_ENV" >&2
  exit 1
fi

export SOURCE_ENV RUNTIME_ENV API_CONTAINER
python3 - <<'PY'
import json
import os
import subprocess
from pathlib import Path


def parse_env(path):
    values = {}
    order = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if value == "${" + key + "}":
            value = os.environ.get(key, "")
        else:
            for env_key, env_value in os.environ.items():
                value = value.replace("${" + env_key + "}", env_value)
        if key not in values:
            order.append(key)
        values[key] = value
    return values, order


source_env = Path(os.environ["SOURCE_ENV"])
runtime_env = Path(os.environ["RUNTIME_ENV"])
source_values, source_order = parse_env(source_env)

merged = {}
order = []
try:
    raw = subprocess.check_output(
        [
            "docker",
            "inspect",
            os.environ["API_CONTAINER"],
            "--format",
            "{{json .Config.Env}}",
        ],
        universal_newlines=True,
        stderr=subprocess.DEVNULL,
    )
    existing = json.loads(raw)
except (subprocess.CalledProcessError, json.JSONDecodeError):
    existing = []

for item in existing:
    if "=" not in item:
        continue
    key, value = item.split("=", 1)
    if key not in merged:
        order.append(key)
    merged[key] = value

for key in source_order:
    if key not in merged:
        order.append(key)
    merged[key] = source_values[key]

mock_mode = merged.get("AGENT_MOCK_MODE", "true").lower() in {"1", "true", "yes", "on"}
if not mock_mode:
    required = ("AI_PROVIDER", "AI_BASE_URL", "AI_MODEL", "AI_API_KEY")
    missing = [key for key in required if not merged.get(key)]
    if missing:
        raise SystemExit("missing required real-model settings: " + ", ".join(missing))
    if merged["AI_API_KEY"] == "${AI_API_KEY}":
        raise SystemExit("AI_API_KEY is still a placeholder")

lines = []
for key in order:
    value = merged[key]
    if "\n" in value or "\r" in value:
        raise SystemExit(f"environment value contains a newline: {key}")
    lines.append(f"{key}={value}")

runtime_env.write_text("\n".join(lines) + "\n", encoding="utf-8")
os.chmod(runtime_env, 0o600)
print(f"[INFO] Runtime env prepared: {runtime_env}")
PY

docker image inspect "$IMAGE_NAME" >/dev/null
docker rm -f "$API_CONTAINER" "$CONSUMER_CONTAINER" >/dev/null 2>&1 || true

docker run -d \
  --restart unless-stopped \
  --name "$API_CONTAINER" \
  --network host \
  --env-file "$RUNTIME_ENV" \
  "$IMAGE_NAME" \
  python -m uvicorn agent_service.main:app --host 0.0.0.0 --port 8000

docker run -d \
  --restart unless-stopped \
  --name "$CONSUMER_CONTAINER" \
  --network host \
  --env-file "$RUNTIME_ENV" \
  "$IMAGE_NAME" \
  python -m agent_service.consumers.run_knowledge_chunk_consumer

for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null; then
    echo "[INFO] API health check passed"
    exit 0
  fi
  sleep 1
done

echo "[ERROR] API health check failed" >&2
docker logs --tail=100 "$API_CONTAINER" >&2 || true
exit 1

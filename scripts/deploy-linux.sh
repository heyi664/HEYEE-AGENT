#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="${APP_DIR:-/root/heyee-agent}"
IMAGE_NAME="${IMAGE_NAME:-heyee-agent:latest}"
API_CONTAINER="${API_CONTAINER:-heyee-agent-api}"
CONSUMER_CONTAINER="${CONSUMER_CONTAINER:-heyee-agent-consumer}"
ENV_FILE="${ENV_FILE:-.env}"
PULL_LATEST="${PULL_LATEST:-true}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
  local exit_code=$?
  printf '\n[ERROR] Deploy failed at line %s, exit code %s.\n' "${BASH_LINENO[0]}" "$exit_code" >&2
  printf '[ERROR] Last command: %s\n' "$BASH_COMMAND" >&2
  printf '\n[INFO] Container status:\n' >&2
  docker ps -a --filter "name=${API_CONTAINER}" --filter "name=${CONSUMER_CONTAINER}" >&2 || true
  printf '\n[INFO] API logs:\n' >&2
  docker logs --tail=80 "$API_CONTAINER" >&2 || true
  printf '\n[INFO] Consumer logs:\n' >&2
  docker logs --tail=80 "$CONSUMER_CONTAINER" >&2 || true
  exit "$exit_code"
}
trap fail ERR

log "Entering project directory: ${APP_DIR}"
cd "$APP_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[ERROR] Env file not found: ${APP_DIR}/${ENV_FILE}" >&2
  exit 1
fi

if [[ ! -f Dockerfile ]]; then
  echo "[ERROR] Dockerfile not found in ${APP_DIR}" >&2
  exit 1
fi

if [[ "$PULL_LATEST" == "true" ]]; then
  log "Git status before pull"
  git status --short

  log "Pulling latest code"
  git pull --ff-only
fi

log "Building Docker image: ${IMAGE_NAME}"
docker build --network=host -t "$IMAGE_NAME" .

log "Recreating application containers"
APP_DIR="$APP_DIR" \
IMAGE_NAME="$IMAGE_NAME" \
API_CONTAINER="$API_CONTAINER" \
CONSUMER_CONTAINER="$CONSUMER_CONTAINER" \
ENV_FILE="$ENV_FILE" \
  bash scripts/recreate-heyee-agent-with-env.sh

log "Checking containers"
docker ps --filter "name=${API_CONTAINER}" --filter "name=${CONSUMER_CONTAINER}"

log "API logs"
docker logs --tail=80 "$API_CONTAINER"

log "Consumer logs"
docker logs --tail=80 "$CONSUMER_CONTAINER"

log "Deploy completed successfully"

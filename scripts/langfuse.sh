#!/usr/bin/env bash
set -euo pipefail

# Start/stop the self-hosted Langfuse stack (docker compose lives outside this repo).
COMPOSE_FILE="${LANGFUSE_COMPOSE_FILE:-$HOME/Documents/SelfStudy/langfuse/docker-compose.yml}"
URL="${LANGFUSE_BASE_URL:-http://localhost:3000}"

compose() { docker compose -f "$COMPOSE_FILE" "$@"; }

case "${1:-}" in
  start)
    compose up -d
    echo "Langfuse starting... UI at $URL (give it ~15s on first boot)"
    ;;
  stop)
    compose down
    echo "Langfuse stopped (data volumes kept)."
    ;;
  restart)
    compose down && compose up -d
    echo "Langfuse restarted. UI at $URL"
    ;;
  status)
    compose ps
    ;;
  logs)
    compose logs -f --tail=100
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}" >&2
    exit 1
    ;;
esac

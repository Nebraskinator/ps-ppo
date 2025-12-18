#!/usr/bin/env bash
set -euo pipefail

PORTS=(8000 8001 8002 8003 8004 8005 8006 8007)
PS_DIR="D:/poke-show-agents/pokemon-showdown"
LOG_DIR="$PS_DIR/logs/multi"
mkdir -p "$LOG_DIR"

pids=()

cleanup() {
  echo "Stopping servers..."
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
}
trap cleanup EXIT INT TERM

echo "Building Pokemon Showdown once..."
(
  cd "$PS_DIR"
  node build
)

for port in "${PORTS[@]}"; do
  echo "Starting PS on :$port"
  (
    cd "$PS_DIR"
    node pokemon-showdown start --no-security --port "$port" \
      >"$LOG_DIR/server_$port.out" 2>"$LOG_DIR/server_$port.err"
  ) &
  pids+=("$!")
  sleep 2
done

echo "Servers running on: ${PORTS[*]}"
echo "PIDs: ${pids[*]}"
wait

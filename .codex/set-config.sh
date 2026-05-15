#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${HOME}/.codex/config.toml"

mkdir -p "$(dirname "$CONFIG_FILE")"

# Create the file if it doesn't exist
touch "$CONFIG_FILE"

# Avoid appending the block more than once
if grep -q '^\[profiles\.llama-local\]' "$CONFIG_FILE"; then
  echo "llama-local profile already exists in: $CONFIG_FILE"
  exit 0
fi

cat >> "$CONFIG_FILE" <<'EOF'

# Local llama-server profile
[profiles.llama-local]
model = "local"
model_provider = "llama-local"
forced_login_method = "api"
model_reasoning_effort = "low"

[model_providers.llama-local]
name = "Local llama-server"
base_url = "http://127.0.0.1:11434/v1"
env_key = "LLAMA_SERVER_API_KEY"
EOF

echo "Appended llama-local profile to: $CONFIG_FILE"
echo "Run with:"
echo "  LLAMA_SERVER_API_KEY=dummy codex --profile llama-local"
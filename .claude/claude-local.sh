export ANTHROPIC_BASE_URL="http://localhost:11434"
export ANTHROPIC_API_KEY="sk-no-key-required"
export ANTHROPIC_AUTH_TOKEN="sk-no-key-required"

# local backends
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_1M_CONTEXT=1
export CLAUDE_CODE_ATTRIBUTION_HEADER=0

claude --model local "$@"
#!/usr/bin/env bash
# Run AgentCore locally for development (no cloud IAM / ECR / S3 deploy permissions needed).
# Use this when your IAM user cannot attach deploy policies (e.g. resonanceUser).

set -euo pipefail

export AGENTCORE_SUPPRESS_RECOMMENDATION="${AGENTCORE_SUPPRESS_RECOMMENDATION:-1}"
export SKIP_IAM_PREFLIGHT=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$ROOT/deploy/bin:$PATH"
chmod +x "$ROOT/deploy/bin/zip" 2>/dev/null || true
if [[ -d "$ROOT/.venv/Scripts" ]]; then
    cp -f "$ROOT/deploy/bin/zip.cmd" "$ROOT/.venv/Scripts/zip.cmd" 2>/dev/null || true
    cp -f "$ROOT/deploy/bin/zip.bat" "$ROOT/.venv/Scripts/zip.bat" 2>/dev/null || true
fi

if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
elif [[ -f "$ROOT/../RAIRA-AI-Research-Assistant/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/../RAIRA-AI-Research-Assistant/.env"
    set +a
fi

NAME="${AGENT_NAME:-rtta_ticket_triage}"
ENTRYPOINT="${ENTRYPOINT:-agentcore_entrypoint.py}"
REGION="${AWS_REGION:-us-east-1}"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-direct_code_deploy}"

agentcore_cmd() {
    PATH="$ROOT/deploy/bin:$ROOT/.venv/Scripts:$PATH" uv run agentcore "$@"
}

configure_args=(
    --name "$NAME"
    --entrypoint "$ENTRYPOINT"
    --deployment-type "$DEPLOYMENT_TYPE"
    --region "$REGION"
    --non-interactive
    --disable-memory
    --language python
)

if [[ "$DEPLOYMENT_TYPE" == "direct_code_deploy" ]]; then
    configure_args+=(--runtime "${PYTHON_RUNTIME:-PYTHON_3_11}")
fi

if [[ ! -f .bedrock_agentcore.yaml ]] && [[ "$DEPLOYMENT_TYPE" == "container" ]]; then
    configure_args=(--create "${configure_args[@]}")
fi

echo "Local AgentCore dev — agent: $NAME"
echo "  (Cloud deploy needs an AWS admin to attach deploy/iam_agentcore_deploy_policy.json)"
echo

agentcore_cmd configure "${configure_args[@]}"

deploy_args=(--agent "$NAME" --local)
[[ -n "${POSTGRES_DSN:-}" ]] && deploy_args+=(--env "POSTGRES_DSN=$POSTGRES_DSN")
[[ -n "${RTTA_MODEL:-}" ]] && deploy_args+=(--env "RTTA_MODEL=$RTTA_MODEL")
[[ -n "${RTTA_EMBEDDINGS:-}" ]] && deploy_args+=(--env "RTTA_EMBEDDINGS=$RTTA_EMBEDDINGS")
[[ -n "${AWS_REGION:-}" ]] && deploy_args+=(--env "AWS_REGION=$AWS_REGION")

echo "Starting local runtime..."
agentcore_cmd deploy "${deploy_args[@]}"

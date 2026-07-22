#!/usr/bin/env bash
# Check AWS IAM permissions for AgentCore deploy and print fix instructions.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_FILE="$ROOT/deploy/iam_agentcore_deploy_policy.json"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32mok\033[0m %s\n" "$*"; }
warn() { printf "  \033[33m!!\033[0m %s\n" "$*"; }
err()  { printf "  \033[31mERR\033[0m %s\n" "$*"; }

bold "Resonance Technologies — AgentCore IAM preflight"
echo

if ! command -v aws >/dev/null 2>&1; then
    err "aws CLI not found. Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

if ! aws sts get-caller-identity >/dev/null 2>&1; then
    err "AWS credentials not configured. Run: aws configure"
    exit 1
fi

if [[ -z "$ACCOUNT_ID" ]]; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
fi
USER_ARN=$(aws sts get-caller-identity --query Arn --output text)
USER_NAME="${USER_ARN##*/}"
ok "account=$ACCOUNT_ID user=$USER_NAME region=$REGION"
echo

missing=0
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-direct_code_deploy}"
DISABLE_MEMORY="${DISABLE_AGENTCORE_MEMORY:-1}"

if [[ "$DISABLE_MEMORY" != "1" ]]; then
    if aws bedrock-agentcore-control list-memories --region "$REGION" --max-results 1 >/dev/null 2>&1; then
        ok "bedrock-agentcore-control:ListMemories"
    else
        warn "bedrock-agentcore-control:ListMemories (or use --disable-memory / DISABLE_AGENTCORE_MEMORY=1)"
        missing=1
    fi
else
    ok "AgentCore memory disabled (DISABLE_AGENTCORE_MEMORY=1)"
fi

if [[ "$DEPLOYMENT_TYPE" == "container" ]]; then
    if aws ecr describe-repositories --region "$REGION" --max-results 1 >/dev/null 2>&1; then
        ok "ecr:DescribeRepositories"
    else
        warn "ecr:DescribeRepositories (required for container deploy)"
        missing=1
    fi
else
    ok "direct_code_deploy selected — ECR not required"
fi

if aws codebuild list-projects --region "$REGION" --max-results 1 >/dev/null 2>&1; then
    ok "codebuild:ListProjects"
elif [[ "$DEPLOYMENT_TYPE" == "container" ]]; then
    warn "codebuild:ListProjects (required for container deploy)"
    missing=1
else
    ok "codebuild:ListProjects (skipped — not required for direct_code_deploy)"
fi

if aws s3api list-buckets --max-items 1 >/dev/null 2>&1; then
    ok "s3:ListAllMyBuckets"
else
    warn "s3:ListAllMyBuckets"
    missing=1
fi

echo
if [[ "$missing" -eq 0 ]]; then
    bold "All preflight checks passed."
    exit 0
fi

bold "Missing IAM permissions"
echo "Your user ($USER_NAME) cannot attach policies to itself."
echo "An AWS account ADMIN must grant deploy permissions."
echo
echo "Option A — AWS Console (admin signs in):"
echo "  1. IAM → Users → $USER_NAME → Permissions"
echo "  2. Add permissions → Create inline policy → JSON"
echo "  3. Paste contents of: deploy/iam_agentcore_deploy_policy.json"
echo "  4. Name: RTTAAgentCoreDeploy → Create policy"
echo
echo "  Console: https://console.aws.amazon.com/iam/home#/users/$USER_NAME"
echo
echo "Option B — AWS CLI (admin credentials, not $USER_NAME):"
echo "  aws iam put-user-policy \\"
echo "    --user-name $USER_NAME \\"
echo "    --policy-name RTTAAgentCoreDeploy \\"
echo "    --policy-document file://$POLICY_FILE"
echo
echo "Option C — Admin creates service-linked roles once (if deploy fails on CreateAgentRuntime):"
echo "  ./deploy/setup_agentcore_slr.sh"
echo
echo "Option D — Develop locally without cloud deploy:"
echo "  ./deploy/deploy_agentcore_local.sh"
echo
echo "After an admin attaches the policy, re-run:"
echo "  ./deploy/deploy_agentcore.sh"
exit 1

#!/usr/bin/env bash
# One-time: create Bedrock AgentCore service-linked roles.
# MUST be run with AWS ADMIN credentials (root / IAM admin), NOT resonanceUser.
#
# Fixes: CreateAgentRuntime "Failed creating service linked role"
#        AccessDenied on iam:CreateServiceLinkedRole

set -euo pipefail

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32mok\033[0m %s\n" "$*"; }
err()  { printf "  \033[31mERR\033[0m %s\n" "$*"; }

bold "Bedrock AgentCore — create service-linked roles"
echo

if ! command -v aws >/dev/null 2>&1; then
    err "aws CLI not found"
    exit 1
fi

CALLER=$(aws sts get-caller-identity --query Arn --output text 2>/dev/null || true)
echo "Caller: $CALLER"
if [[ "$CALLER" == *":user/resonanceUser" ]]; then
    err "You are signed in as resonanceUser — this user cannot create service-linked roles."
    err "Sign in as an AWS account ADMIN, then re-run this script."
    echo
    echo "Or in IAM Console as admin:"
    echo "  1. Attach BedrockAgentCoreFullAccess to resonanceUser"
    echo "  2. OR replace RTTAAgentCoreDeploy JSON with deploy/iam_agentcore_deploy_policy.json"
    echo "  3. Then as admin run the create-service-linked-role commands below"
    exit 1
fi

# Role needed by CreateAgentRuntime (most common failure)
create_one() {
    local service="$1"
    local role="$2"
    if aws iam get-role --role-name "$role" >/dev/null 2>&1; then
        ok "already exists: $role"
        return 0
    fi
    if aws iam create-service-linked-role --aws-service-name "$service" >/dev/null; then
        ok "created: $role"
    else
        err "failed: $service"
        return 1
    fi
}

create_one "runtime-identity.bedrock-agentcore.amazonaws.com" \
    "AWSServiceRoleForBedrockAgentCoreRuntimeIdentity"

create_one "network.bedrock-agentcore.amazonaws.com" \
    "AWSServiceRoleForBedrockAgentCoreNetwork" || true

create_one "bedrock-agentcore.amazonaws.com" \
    "AWSServiceRoleForBedrockAgentCoreGatewayNetwork" || true

create_one "identity-network.bedrock-agentcore.amazonaws.com" \
    "AWSServiceRoleForBedrockAgentCoreIdentity" || true

echo
bold "Done. Switch back to resonanceUser and run:"
echo "  ./deploy/deploy_agentcore.sh"

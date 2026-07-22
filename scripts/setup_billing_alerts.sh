#!/usr/bin/env bash
set -euo pipefail
EMAIL="${ALERT_EMAIL:?set ALERT_EMAIL}"

# AWS budget alert at $10/day
aws budgets create-budget --account-id "$(aws sts get-caller-identity --query Account --output text)" \
    --budget '{"BudgetName":"rtta-lab-daily","BudgetLimit":{"Amount":"10","Unit":"USD"},"TimeUnit":"DAILY","BudgetType":"COST"}' \
    --notifications-with-subscribers '[{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":100},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"'"$EMAIL"'"}]}]'

# GCP budget alert at $10/day
BILLING_ACCOUNT="$(gcloud billing accounts list --format='value(name)' --limit=1)"
gcloud billing budgets create \
    --billing-account="$BILLING_ACCOUNT" \
    --display-name="rtta-lab-daily" \
    --budget-amount="10USD" \
    --threshold-rule=percent=1.0,basis=current-spend \
    --calendar-period=daily
echo "Alerts set up. Watch $EMAIL."

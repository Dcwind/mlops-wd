#!/usr/bin/env bash
set -euo pipefail

# Spin up LocalStack
docker compose up -d localstack

sleep 5  # Wait for LocalStack to start

# Always clean up, even on Ctrl-C
cleanup() {
  docker compose down localstack
}
trap cleanup EXIT

# Local fake AWS creds
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=ap-southeast-1
export AWS_ENDPOINT_URL=http://localhost:4566

# Create bucket if it is not there yet
pipenv run aws --endpoint-url=$AWS_ENDPOINT_URL s3 mb s3://test-bucket || true
pipenv run aws --endpoint-url=$AWS_ENDPOINT_URL s3 ls

# Run the Python integration test
export BUCKET=test-bucket
pipenv run python integration_test.py

# Check if the test passed
if [ $? -ne 0 ]; then
  echo "Integration test failed."
  exit 1
else
  echo "Integration test passed."
fi  
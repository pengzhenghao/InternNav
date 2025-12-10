#!/bin/bash

 # cd to the repo root (assuming this script is run from agent/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Basic colors for pretty output
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
BOLD="\033[1m"
RESET="\033[0m"

echo -e "${BOLD}${BLUE}Running from:${RESET} $REPO_ROOT"

# Configuration
MOCK_MODE=true
GOAL="Find the fire extinguisher"

# Function to cleanup background processes on exit
cleanup() {
    echo -e "${YELLOW}[CLEANUP] Stopping all services...${RESET}"
    kill $(jobs -p) 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM

echo -e "${BOLD}==============================================${RESET}"
echo -e "${BOLD}   Agentic Navigation System (System 3 Demo)${RESET}"
echo -e "${BOLD}==============================================${RESET}"
echo -e "${BOLD}Configuration:${RESET}"
echo "  MOCK_MODE = $MOCK_MODE"
echo "  GOAL      = $GOAL"
echo

# 1. Start VLLM (Brain)
if [ "$MOCK_MODE" = true ]; then
    echo -e "${GREEN}[1/3] Starting Mock VLLM Brain...${RESET}"
    python3 agent/mock_vllm.py > agent/vllm.log 2>&1 &
    VLLM_PID=$!
    echo "      Logs: agent/vllm.log (tail -f agent/vllm.log)"
else
    echo -e "${GREEN}[1/3] Assuming real VLLM is running at localhost:8000...${RESET}"
    # If you have a start script for VLLM, put it here
fi

# 2. Start Robot System (System 2)
# Ideally this is the real grpc_internvla_server.py
# For this demo, if we don't have the GPU/Checkpoints loaded, 
# we can use a simple mock for the robot server too.
# Let's check if we can run the real one, or fallback to a simple mock.

if [ "$MOCK_MODE" = true ]; then
    echo -e "${GREEN}[2/3] Starting Mock Robot Server (System 2)...${RESET}"
    # We will create a tiny mock robot server inline here for convenience or separate file
    python3 agent/mock_robot.py > agent/robot.log 2>&1 &
    ROBOT_PID=$!
    echo "      Logs: agent/robot.log (tail -f agent/robot.log)"
else
    echo -e "${GREEN}[2/3] Starting Real InternVLA Server...${RESET}"
    # cd scripts/realworld && python grpc_internvla_server.py ...
fi

# Wait for services to spin up
echo -e "${YELLOW}Waiting 5 seconds for services to come up...${RESET}"
sleep 5

# 3. Start Agent (System 3)
echo -e "${GREEN}[3/3] Starting Agentic Controller...${RESET}"
echo -e "${BOLD}Goal:${RESET} $GOAL"
echo -e "${BLUE}Agent logs will stream below (from Python logging)...${RESET}"
python3 agent/agent_server.py --goal "$GOAL" --interval 3

# Keep script running
wait

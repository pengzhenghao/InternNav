import time
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MockVLLM")

# Scenario state
step_count = 0

@app.route("/v1/chat/completions", methods=['POST'])
def chat_completions():
    global step_count
    data = request.json
    messages = data.get('messages', [])
    logger.info(f"Received request with {len(messages)} messages")
    
    # Simple scripted scenario for testing
    step_count += 1
    
    if step_count <= 2:
        # Phase 1: Search
        content = """{
  "thought": "I see an office desk. The goal is 'Find fire extinguisher'. I don't see it here. I see a door in the background.",
  "status": "SEARCHING",
  "instruction": "Go to the door"
}"""
    elif step_count <= 4:
        # Phase 2: Navigate
        content = """{
  "thought": "I am at the door. I see a long hallway. I should check down the hall.",
  "status": "NAVIGATING",
  "instruction": "Walk down the hallway"
}"""
    else:
        # Phase 3: Found
        content = """{
  "thought": "I see a red object on the wall. That is the fire extinguisher.",
  "status": "DONE",
  "instruction": "STOP"
}"""

    # Mimic OpenAI Response Format
    response = {
        "id": "chatcmpl-mock-123",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }
    
    return jsonify(response)

@app.route("/v1/models", methods=['GET'])
def list_models():
    return jsonify({
        "object": "list",
        "data": [{
            "id": "mock-model",
            "object": "model",
            "created": 1677610602,
            "owned_by": "mock"
        }]
    })

if __name__ == "__main__":
    print("Starting Mock VLLM on port 8000...")
    app.run(host='0.0.0.0', port=8000)

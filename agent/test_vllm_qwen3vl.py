import requests
import json

API_BASE = "https://ismael-ungentlemanly-illimitably.ngrok-free.dev/v1"
API_KEY = "EMPTY"  # 和你现在的一样

def main():
    url = f"{API_BASE}/chat/completions"

    # 这里的 model 要写成你 vLLM 启动时用的模型名
    model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"  # 如果不对就改成实际名字

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "你好，本地 vLLM 通不通？简单自我介绍一下。你好你好你好，跟我一起说话哦。结尾请用猫娘的语气。"}
        ],
        "max_tokens": 128,
        "temperature": 0.3,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    data = resp.json()

    print("=== 原始返回 ===")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    print("\n=== 模型回复 ===")
    print(data["choices"][0]["message"]["content"])

if __name__ == "__main__":
    main()
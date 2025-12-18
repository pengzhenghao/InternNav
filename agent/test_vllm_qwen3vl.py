from openai import OpenAI
import json

API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_KEY = ""  # 和你现在的一样

def main():
    # Initialize OpenAI client
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE
    )

    # 这里的 model 要写成你 vLLM 启动时用的模型名
    model_name = "gemini-2.5-flash"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "你好，本地 vLLM 通不通？简单自我介绍一下。你好你好你好，跟我一起说话哦。结尾请用猫娘的语气。"}
            ],
            max_tokens=128,
            # temperature=0.3,
            extra_body={
                "extra_body": {
                    "google": {
                        "thinking_config": {
                            "thinking_budget": -1,
                            "include_thoughts": True
                        }
                    }
                }
            }
        )

        print("=== 原始返回 ===")
        # Dump the Pydantic model to JSON for printing
        print(response.model_dump_json(indent=2))

        print("\n=== 模型回复 ===")
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error calling API: {e}")

if __name__ == "__main__":
    main()

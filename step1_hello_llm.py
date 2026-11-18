"""
Step 1：跑通第一个模型 API 调用 —— 流式输出 + 参数实验
对应用户学习第一天（60 分钟）的核心动手环节。

运行前：
  1. pip install -r requirements.txt
  2. cp .env.example .env 并填入 API Key
  3. python step1_hello_llm.py
"""
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
)
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")

SYSTEM_PROMPT = """你是一位资深的 Go 后端工程师兼技术面试官。
回答简洁、准确，必要时给出代码示例。用中文回答。"""


def ask_stream(question: str, temperature: float = 0.7, top_p: float = 1.0) -> str:
    """发起一次流式对话，逐字打印并拼接完整回答。"""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )

    parts = []
    print("\n[回答] ", end="", flush=True)
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)
            print(delta, end="", flush=True)
    print("\n")
    return "".join(parts)


def main():
    if not os.getenv("OPENAI_API_KEY") or "你的key" in os.getenv("OPENAI_API_KEY", ""):
        print("请先复制 .env.example 为 .env，并填入真实的 API Key。")
        sys.exit(1)

    question = sys.argv[1] if len(sys.argv) > 1 else "用三句话解释 Go 的 goroutine 和线程的区别。"

    print(f"[模型] {MODEL}")
    print(f"[问题] {question}")
    ask_stream(question)

    print("[小实验] 把 temperature 改成 0.0 再跑一次，对比回答的稳定性和创造性：")
    ask_stream(question, temperature=0.0)


if __name__ == "__main__":
    main()

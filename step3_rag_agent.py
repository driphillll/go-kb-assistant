"""
Step 3：RAG 版 Agent —— 工具 + 规划 + 记忆 + 真检索

与 Step2 的区别只有一处：
  Step2 的 get_go_doc 是"字典精确查表"（GO_DOCS[symbol]）。
  Step3 的 get_go_doc 改成"语义检索"（rag_core.rag_search）。

其它全部复用 Step2：
  - TOOLS 工具描述（给模型看的 JSON Schema）不变
  - run_agent 主循环（规划 -> 执行工具 -> 观察 -> 再规划）不变
  - messages 记忆 不变
  - tool_choice / extra_body 配置 不变

运行前：
  .env 配好本地 Ollama；nomic-embed-text 已拉取（ollama pull nomic-embed-text）
  首次运行会自动根据 corpus/ 建 rag_index.json

运行：
  .venv\\Scripts\\python.exe step3_rag_agent.py
  .venv\\Scripts\\python.exe step3_rag_agent.py "Go 的 channel 和 goroutine 有什么区别？"
"""
import os
import sys

from openai import OpenAI
from dotenv import load_dotenv

from rag_core import rag_search, build_index

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
)
STEP3_MODEL = os.getenv("STEP3_MODEL", os.getenv("STEP2_MODEL", "qwen3:8b"))
MAX_STEPS = 8

# 启动时确保索引就绪（首次会向量化，稍慢）
_INDEX = build_index()


# ============================================================
# 第一部分：工具实现
# 唯一变化点：get_go_doc 内部从字典查表换成 RAG 检索。
# 注意函数签名 (query) 和返回类型 (str) 与 Step2 保持一致——
# 这就是"接口不变"，所以下面 TOOLS 和 run_agent 一个字都不用改。
# ============================================================
def get_go_doc(query: str) -> str:
    """按语义检索 Go 知识库，返回最相关的原文片段（含来源）。"""
    print(f"      [工具执行] get_go_doc(query={query!r})")
    return rag_search(query, index=_INDEX)


def list_symbols() -> str:
    """列出知识库里可查询的主题。"""
    print("      [工具执行] list_symbols()")
    from rag_core import load_and_split
    names = sorted({c["section"] for c in load_and_split()})
    return "可查询主题：" + ", ".join(names)


TOOL_IMPL = {
    "get_go_doc": get_go_doc,
    "list_symbols": list_symbols,
}


# ============================================================
# 第二部分：工具的 JSON Schema —— 与 Step2 相同
# 只把参数名从 symbol 改成 query、描述改成"语义检索"，更贴合新实现。
# ============================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_go_doc",
            "description": "用自然语言检索 Go 知识库，返回最相关的原文片段和来源。回答 Go 相关问题时优先调用。支持概念性问题，如'channel 如何做同步'、'GMP 是什么'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的问题或主题描述"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_symbols",
            "description": "列出知识库里所有可查询的 Go 主题。不确定有哪些内容可查时调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ============================================================
# 第三部分：System Prompt —— 增加"引用来源"的要求
# 这是 RAG 相对 Step2 的新增价值：回答必须基于检索片段，并标注来源。
# ============================================================
SYSTEM_PROMPT = """你是一位资深的 Go 后端工程师，拥有一个 Go 知识库可以检索。
回答问题时必须遵守：
1. 先调用 get_go_doc 工具检索相关资料，基于工具返回的原文片段作答，不要凭记忆编造。
2. 如果不确定知识库里有哪些主题，先调用 list_symbols。
3. 可以多次检索以收集齐资料，再综合给出最终答案。
4. 最终答案用中文简洁回答，并在结尾用一行标注信息来源（格式：来源：文件名 → 小节标题）。只引用工具实际返回的片段来源。"""


# ============================================================
# 第四部分：Agent 主循环 —— 与 Step2 完全一致，无需改动
# ============================================================
def run_agent(question: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    print(f"[模型] {STEP3_MODEL}")
    print(f"[问题] {question}\n")

    for step in range(1, MAX_STEPS + 1):
        print(f"  --- 第 {step} 轮 ---")
        response = client.chat.completions.create(
            model=STEP3_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        msg = response.choices[0].message
        messages.append(msg)

        # 情况 A：模型不再请求工具 -> 输出最终答案
        if not msg.tool_calls:
            print("\n[最终答案]\n" + (msg.content or "(空)"))
            return msg.content or ""

        # 情况 B：执行工具，把结果回填进记忆
        for call in msg.tool_calls:
            name = call.function.name
            import json
            args = json.loads(call.function.arguments or "{}")
            print(f"      [模型决策] 调用工具 {name}，参数 {args}")
            result = TOOL_IMPL[name](**args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    print(f"\n达到最大步数 {MAX_STEPS}，强制停止。")
    return ""


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "channel 是怎么做同步的？和 goroutine 有什么区别？"
    run_agent(question)

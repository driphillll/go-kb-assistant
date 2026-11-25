"""
Step 4：状态机式 Agent —— 给 RAG 工具链加"人工确认中断点"

与 Step3 的关系：
  Step3：模型决定调工具 -> 立刻执行 -> 回填结果 -> 再规划（全自动，人只能等结果）
  Step4：模型决定调工具 -> 【暂停，人审核/改参数】-> 执行 -> 回填 -> 再规划

唯一新增的核心概念：人工确认中断点（Human-in-the-loop interrupt）。
这正是总纲阶段二的验收点之一："加入人工确认中断点"。它适合工具带副作用
（写库、发请求、花钱）的场景——把"模型想做什么"和"真的去做"拆成两步。

为什么这里不引入 LangGraph：
  项目 step1/2/3 全程用原生 OpenAI SDK + 手写循环，风格连贯、你能读懂每一行。
  LangGraph 的 interrupt_before=["tools"] 本质就是本脚本 CONFIRM 节点做的事。
  建议先用原生版吃透"状态机 + 中断点"的概念，等阶段二要接 LangFuse 追踪、
  要做复杂分支时再迁移到 LangGraph，那时是给现有节点换个壳，不是重写。

状态机结构（每一步都打印状态名，方便你观察流转）：
                    ┌──────────────────────────────────┐
                    │                                  │
                    v                                  │
  START -> [PLAN 模型规划] --无 tool_calls--> [ANSWER] -> END
                │ 有 tool_calls                          ^
                v                                       │
         [CONFIRM 人工确认中断点]                        │
           y执行 / n拒绝 / e改参数 / q退出               │
                │ y 或 e                                │
                v                                       │
         [EXECUTE 执行工具] -> [OBSERVE 回填结果] ───────┘

运行：
  .venv\\Scripts\\python.exe step4_stateful_agent.py
  .venv\\Scripts\\python.exe step4_stateful_agent.py "channel 是怎么同步的？"
"""

import json
import os
import sys

from openai import OpenAI
from dotenv import load_dotenv

# 复用 Step3 已建好的真实检索能力（top_k 默认形参就是你在 rag_core 里加的那个）
from rag_core import rag_search, build_index

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
)
STEP4_MODEL = os.getenv("STEP4_MODEL", os.getenv("STEP3_MODEL", "qwen3:8b"))
MAX_STEPS = 8

# 启动时确保索引就绪（索引已存在则秒回）
_INDEX = build_index()


# ============================================================
# 第一部分：工具实现 —— 与 Step3 相同，但把 top_k 暴露给模型决定
# Step3 里 top_k 用默认值 3；Step4 让模型按问题自己选，这是你的新发现的实际用法。
# ============================================================
def get_go_doc(query: str, top_k: int = 3) -> str:
    """按语义检索 Go 知识库，返回最相关的原文片段（含来源）。"""
    print(f"      [工具执行] get_go_doc(query={query!r}, top_k={top_k})")
    return rag_search(query, top_k=top_k, index=_INDEX)


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

# 工具 JSON Schema：新增 top_k 参数（默认 3），对应你发现的默认形参
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_go_doc",
            "description": "用自然语言检索 Go 知识库，返回最相关的原文片段和来源。回答 Go 相关问题时优先调用。支持概念性问题，如'channel 如何做同步'、'GMP 是什么'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的问题或主题描述"},
                    "top_k": {
                        "type": "integer",
                        "description": "返回最相关的几条片段，默认 3；问题宽泛可调到 5",
                        "default": 3,
                    },
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

SYSTEM_PROMPT = """你是一位资深的 Go 后端工程师，拥有一个 Go 知识库可以检索。
回答问题时必须遵守：
1. 先调用 get_go_doc 工具检索相关资料，基于工具返回的原文片段作答，不要凭记忆编造。
2. 如果不确定知识库里有哪些主题，先调用 list_symbols。
3. 问题宽泛时可把 get_go_doc 的 top_k 调到 5，收集齐资料再综合回答。
4. 最终答案用中文简洁回答，并在结尾用一行标注信息来源（格式：来源：文件名 → 小节标题）。只引用工具实际返回的片段来源。"""


# ============================================================
# 第二部分：CONFIRM 节点 —— 人工确认中断点（本 Step 唯一的新东西）
# 返回处理后的 tool_calls 列表；返回 None 表示用户退出。
# 拒绝的调用也会保留，但会被替换成"用户拒绝"的回填结果，让模型知道并调整。
# ============================================================
def confirm_tool_calls(tool_calls: list) -> list | None:
    if not tool_calls:
        return tool_calls

    print("\n" + "  " + "=" * 56)
    print("  [CONFIRM 人工确认中断点] 模型请求执行以下工具：")
    for i, call in enumerate(tool_calls, 1):
        fn = call["function"]
        print(f"    {i}. {fn['name']}({fn['arguments']})")
    print("  " + "-" * 56)
    print("    y = 全部执行   n = 全部拒绝   e = 逐个审阅/改参数   q = 退出")
    print("  " + "=" * 56)

    choice = input("  你的决定 [y/n/e/q，默认 y]: ").strip().lower() or "y"

    if choice == "q":
        print("  用户中止。")
        return None
    if choice == "n":
        for call in tool_calls:
            call["_rejected"] = True  # 打标记，EXECUTE 阶段回填"用户拒绝"
        print("  已拒绝全部工具调用。")
        return tool_calls
    if choice == "e":
        for i, call in enumerate(tool_calls, 1):
            fn = call["function"]
            print(f"\n    第 {i} 个：{fn['name']}({fn['arguments']})")
            sub = input("    y=执行 / n=拒绝 / 直接输入新参数JSON(改后执行) [y]: ").strip().lower() or "y"
            if sub == "n":
                call["_rejected"] = True
            elif sub not in ("y", ""):
                try:
                    new_args = json.loads(sub)
                    fn["arguments"] = json.dumps(new_args, ensure_ascii=False)
                    print(f"    已改为：{fn['arguments']}")
                except json.JSONDecodeError:
                    print("    不是合法 JSON，保持原参数。")
        return tool_calls
    return tool_calls  # y：原样执行


# ============================================================
# 第三部分：状态机主循环 —— PLAN -> CONFIRM -> EXECUTE -> OBSERVE -> 回到 PLAN
# ============================================================
def run_agent(question: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    print(f"[模型] {STEP4_MODEL}")
    print(f"[问题] {question}\n")

    for step in range(1, MAX_STEPS + 1):
        # ---------- 状态：PLAN ----------
        print(f"  --- 第 {step} 轮 · [PLAN 模型规划] ---")
        response = client.chat.completions.create(
            model=STEP4_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        msg = response.choices[0].message

        # 把 assistant 消息转成普通 dict，方便在中断点编辑参数
        assistant_msg = {
            "role": "assistant",
            "content": msg.content,
        }
        raw_calls = []
        if msg.tool_calls:
            for call in msg.tool_calls:
                raw_calls.append({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments or "{}",
                    },
                })
            assistant_msg["tool_calls"] = raw_calls
        messages.append(assistant_msg)

        # ---------- 无工具调用 -> ANSWER ----------
        if not raw_calls:
            print(f"\n  [ANSWER 最终答案]\n")
            print(msg.content or "(空)")
            return msg.content or ""

        # ---------- 状态：CONFIRM 人工确认中断点 ----------
        decided = confirm_tool_calls(raw_calls)
        if decided is None:
            return "(用户中止)"

        # ---------- 状态：EXECUTE + OBSERVE ----------
        print(f"  --- 第 {step} 轮 · [EXECUTE 执行 / OBSERVE 回填] ---")
        for call in decided:
            if call.get("_rejected"):
                result = "用户拒绝执行该工具调用。请根据已有信息回答，或换一种方式检索。"
                print(f"      [已拒绝] {call['function']['name']}")
            else:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"] or "{}")
                print(f"      [模型决策] {name} {args}")
                result = TOOL_IMPL[name](**args)
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            })

    print(f"\n达到最大步数 {MAX_STEPS}，强制停止。")
    return ""


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "channel 是怎么做同步的？和 goroutine 有什么区别？"
    run_agent(question)

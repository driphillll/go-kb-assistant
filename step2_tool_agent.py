"""
Step 2：最小可运行 Agent —— 工具调用 + 规划 + 记忆

在 step1（模型能聊天）的基础上，让模型学会"自己动手查资料再回答"。
一个 Agent 的最小闭环就是三件事：
  1. 工具（Tools）：模型能调用的外部能力，这里先查内置的 Go 符号文档。
  2. 规划（Planning）：模型根据问题决定"要不要查、查哪个、查几次"。
  3. 记忆（Memory）：messages 列表保存完整对话与每轮工具结果，作为下一步的依据。

运行前：
  .env 里配好本地 Ollama（OPENAI_BASE_URL=http://localhost:11434/v1，KEY=ollama）
  本步推荐 MODEL_NAME 用 qwen3:8b（tools 能力更稳），可用环境变量 STEP2_MODEL 覆盖。

运行：
  .venv\\Scripts\\python.exe step2_tool_agent.py
  .venv\\Scripts\\python.exe step2_tool_agent.py "Go 的 channel 和 goroutine 有什么区别？"
"""
import json
import os
import sys

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
)
STEP2_MODEL = os.getenv("STEP2_MODEL", "qwen3:8b")
MAX_STEPS = 8  # 防止模型在工具循环里停不下来


# ============================================================
# 第一部分：工具的"真实实现"
# 这里用内置字典模拟一个 Go 知识库。Step 3 会把 get_go_doc
# 换成真正的 RAG 检索（向量库 + 重排序），调用方式完全不变。
# ============================================================
GO_DOCS = {
    "goroutine": (
        "goroutine 是 Go 运行时管理的轻量级线程。由 go 关键字启动，"
        "初始栈只有几 KB，可动态增长。调度由 Go 的 GMP 模型完成，"
        "切换开销远小于操作系统线程。适合高并发 I/O 与并行任务。"
    ),
    "channel": (
        "channel 是 goroutine 之间通信和同步的管道，类型化且线程安全。"
        "分为无缓冲（同步，发送接收必须同时就绪）和有缓冲（异步，缓冲区满才阻塞）。"
        "支持 for-range 遍历、close 关闭、select 多路复用。核心理念："
        "不要通过共享内存来通信，而要通过通信来共享内存。"
    ),
    "sync.Mutex": (
        "sync.Mutex 是互斥锁，用于保护临界区，同一时刻只允许一个 goroutine 访问。"
        "Lock() 加锁，Unlock() 解锁，必须成对出现，通常用 defer 保证释放。"
        "Go 1.18+ 的 sync.Mutex 支持 starvation 模式，避免长期等待的 goroutine 饿死。"
    ),
    "interface": (
        "interface 是 Go 的抽象类型，定义一组方法签名。任何实现了这些方法的类型"
        "都隐式满足该接口（鸭子类型），无需显式声明。空接口 interface{} "
        "可持有任意类型值，底层由 type + value 两部分组成（iface / eface）。"
    ),
    "defer": (
        "defer 把函数调用推迟到当前函数返回前执行，常用于资源释放。"
        "多个 defer 按后进先出（LIFO）顺序执行。defer 的参数在注册时求值，"
        "但函数体在返回时才执行。常与 recover 配合做 panic 恢复。"
    ),
}


def get_go_doc(symbol: str) -> str:
    """查询 Go 符号的定义文档。symbol 如 goroutine / channel / sync.Mutex。"""
    print(f"      [工具执行] get_go_doc(symbol={symbol!r})")
    if symbol in GO_DOCS:
        return GO_DOCS[symbol]
    available = ", ".join(GO_DOCS.keys())
    return f"未找到符号 {symbol!r}。可用符号：{available}"


def list_symbols() -> str:
    """列出当前知识库中所有可查询的 Go 符号。"""
    print("      [工具执行] list_symbols()")
    return "可查询符号：" + ", ".join(GO_DOCS.keys())


# 工具名 -> 真实函数的映射，供 Agent 执行时查找
TOOL_IMPL = {
    "get_go_doc": get_go_doc,
    "list_symbols": list_symbols,
}


# ============================================================
# 第二部分：工具的 JSON Schema（描述给模型看，告诉它有哪些工具、怎么传参）
# OpenAI 兼容格式：type=function，function 里给名字、说明、参数结构。
# ============================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_go_doc",
            "description": "查询一个 Go 符号（如 goroutine、channel、sync.Mutex）的定义文档。回答关于 Go 概念的问题时应优先调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "要查询的 Go 符号名"}
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_symbols",
            "description": "列出知识库里所有可查询的 Go 符号。当不确定有哪些符号可查时调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ============================================================
# 第三部分：System Prompt —— 这是"规划"的行为约束，也是今天要调试的对象
# ============================================================
SYSTEM_PROMPT = """你是一位资深的 Go 后端工程师。
回答关于 Go 的问题时，必须遵守：
1. 先判断问题涉及哪些 Go 符号，然后调用 get_go_doc 工具获取准确资料，不要凭记忆作答。
2. 如果不确定有哪些符号可查，先调用 list_symbols。
3. 可以多次调用工具收集齐资料后，再综合给出最终答案。
4. 最终答案基于工具返回的内容，用中文简洁回答，并注明信息来自工具查询。"""


# ============================================================
# 第四部分：Agent 主循环 —— 规划 → 执行工具 → 观察 → 再规划，直到给出答案
# ============================================================
def run_agent(question: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    # "记忆"：messages 保存完整上下文。系统指令 + 用户问题 + 每轮助手消息 + 每次工具结果。
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    print(f"[模型] {STEP2_MODEL}")
    print(f"[问题] {question}\n")

    for step in range(1, MAX_STEPS + 1):
        print(f"  --- 第 {step} 轮 ---")
        response = client.chat.completions.create(
            model=STEP2_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",  # 让模型自己决定调不调工具
            # qwen3 默认开启思考会变慢且输出 reasoning，这里显式关闭以获得干净的工具调用
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        msg = response.choices[0].message

        # 把助手这一轮（可能含 tool_calls）存入记忆
        messages.append(msg)

        # 情况 A：模型不再请求工具 —— 说明它已收集齐信息，输出最终答案
        if not msg.tool_calls:
            print("\n[最终答案]\n" + (msg.content or "(空)"))
            return msg.content or ""

        # 情况 B：模型请求调用工具 —— 逐个执行，并把结果回填进记忆
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            print(f"      [模型决策] 调用工具 {name}，参数 {args}")
            result = TOOL_IMPL[name](**args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )

    print(f"\n达到最大步数 {MAX_STEPS}，强制停止。")
    return ""


# ============================================================
# 第五部分：Prompt 调试小实验 —— 对比"有/无系统指令"时模型是否会主动用工具
# 这是今天的 Prompt 实验：看约束性 system prompt 是否提升了工具调用率。
# ============================================================
def prompt_experiment(question: str) -> None:
    weak_prompt = "你是一个助手，用中文回答问题。"
    print("\n" + "=" * 56)
    print("Prompt 对比实验：弱提示 vs 强提示，观察是否主动调用工具")
    print("=" * 56)
    print("\n【A】弱提示（未强制用工具）")
    run_agent(question, system_prompt=weak_prompt)
    print("\n【B】强提示（强制先查工具）")
    run_agent(question, system_prompt=SYSTEM_PROMPT)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "Go 的 channel 和 goroutine 有什么区别？"

    run_agent(question)

    # 想跑 Prompt 对比实验时，把下面这行的注释去掉：
    # prompt_experiment(question)

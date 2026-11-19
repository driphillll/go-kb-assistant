# Go 项目知识库助手

一个基于 RAG 的问答系统：把 Go 项目的源码、README、文档切分入库，然后用自然语言提问，得到带引用溯源的答案。

## 为什么选这个题目

- **领域护城河**：你精通 Go，能设计出真正懂 Go 的切分规则（按 package / 函数 / 方法切，而不是盲目按字符数切），这是纯 AI 背景的人做不出来的。
- **覆盖高频考点**：RAG 全链路（解析 → 切分 → Embedding → 向量库 → 检索 → 重排序 → 引用）几乎出现在每一条 Agent 岗 JD 里。
- **可演进成 Agent**：知识库助手是 L3，后期接入工具调用（读文件、跑 go doc、查符号定义）就升级成 L2+L4 的 Agent，一个项目贯穿 12 周。

## 12 周演进路线

| 阶段 | 周数 | 本项目要做到什么 |
|------|------|------------------|
| 一 | W1–2 | 跑通模型 API + 流式输出（本仓库起点） |
| 二 | W3–5 | 给助手加工具：查符号定义、读文件内容 |
| 三 | W6–8 | 完整 RAG：Go 源码切分 + 向量库 + 重排序 + 引用溯源 + RAGAS 评测 |
| 四 | W9–12 | 多智能体：规划 Agent + 检索 Agent + 审查 Agent；MCP Server；Docker 部署 |

## 快速开始

学习阶段默认用本机 Ollama（免费）。`.env.example` 已配好，模型名按本机实际安装的填（`qwen3:4b` / `qwen3:8b`）。

**通用（Linux / macOS）：**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python step1_hello_llm.py
```

**Windows（本机已验证，注意 `python` 被商店别名劫持，用 `py` 启动器）：**
```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
copy .env.example .env
.venv\Scripts\python.exe step1_hello_llm.py
```
> 在 PyCharm 里：Settings → Project → Python Interpreter → Add Interpreter → 选择本目录 `.venv\Scripts\python.exe` 即可。PyCharm 本身不创造隔离，它只是选用这个 venv。

想切换到 DeepSeek 时，把 `.env` 里的三行按 `.env.example` 中「方案 B」注释替换即可，代码不用改——这就是 OpenAI 兼容接口的好处。

> 提示：qwen3 是思考模型，回答前会先内部推理，等待几秒属正常。课程后期做工具调用时，`qwen3:8b` 的 `tools` 能力更稳，可优先用它。

## 当前进度

- [x] Step 1：模型 API 调用 + 流式输出（`step1_hello_llm.py`）
- [x] Step 2：工具调用 + 最小可运行 Agent（`step2_tool_agent.py`）
  - 三件套：工具（`get_go_doc` / `list_symbols`）+ 规划（工具调用循环）+ 记忆（`messages` 历史）
  - 内置 Go 符号文档作为数据源，Step 3 将替换为真正的 RAG 检索
  - 含 Prompt 对比实验（`prompt_experiment`，弱提示 vs 强提示的工具调用率）
- [x] Step 3：RAG 检索基础版（`rag_core.py` + `step3_rag_agent.py`）
  - `corpus/go_kb.md`：7 个主题的结构化 Go 知识语料（含 Step2 字典里没有的 context / GMP，证明语义检索强于硬编码查表）
  - `rag_core.py`：切分（按标题）→ embedding（Ollama nomic-embed-text）→ JSON 索引 → 余弦相似度检索，指纹变化才重建
  - `step3_rag_agent.py`：**复用 Step2 主循环一字未改，只把 `get_go_doc` 内部从字典查表换成语义检索**；答案强制带来源
  - 端到端实测：模型自主规划 2 轮检索（同步机制 + 两者区别），答案基于片段并标注来源
  - 前置：`ollama pull nomic-embed-text`；`EMBED_MODEL` / `STEP3_MODEL` 可用环境变量覆盖
  - 待进阶：真实 Go 源码切分（按 package/函数）、ChromaDB/FAISS 向量库、rerank 重排序、RAGAS 评测
- [ ] Step 4：多智能体 + 部署

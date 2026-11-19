"""
Step 3 - RAG 核心模块（检索侧）

职责：把 corpus/ 下的 Go 知识文档变成"可检索的知识库"，并提供一个
       rag_search(question) 函数，供 step3 的 Agent 工具 get_go_doc 调用。

设计原则（对齐 Step2 的"接口不变"思想）：
  - 本模块只负责"给定问题 -> 返回最相关的原文片段 + 来源"。
  - 至于这些片段怎么被模型用、Agent 循环怎么走，交给 step3_rag_agent.py，
    那部分代码与 Step2 完全一致，不用改。

RAG 全链路的四个环节在这里都能看到：
  1) 解析/切分(load + split)  2) 向量化(embed)  3) 入库(build)  4) 检索(search)

运行：
  .venv\\Scripts\\python.exe rag_core.py            # 建库并跑一个示例检索
  .venv\\Scripts\\python.exe rag_core.py --rebuild  # 强制重建索引
"""
import os
import re
import sys
import json
import hashlib
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
CORPUS_DIR = BASE_DIR / "corpus"
INDEX_FILE = BASE_DIR / "rag_index.json"          # 简易持久化：存 chunk + 向量
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
)


# ------------------------------------------------------------
# 环节 1：加载 + 切分
# 按 markdown 的二级标题(##)切块，每块带上"来源文件 + 小节标题"元信息。
# 为什么按标题切而不是按字数硬切？因为标题是天然的语义边界——这是你
# 作为 Go 工程师的领域知识能派上用场的地方（真实项目里可按 package/函数切）。
# ------------------------------------------------------------
def load_and_split(corpus_dir=CORPUS_DIR):
    chunks = []
    for md in sorted(Path(corpus_dir).glob("*.md")):
        text = md.read_text(encoding="utf-8")
        # 先按一级标题(#)切成"主题块"，主题名作为归属；再在每个主题块内按
        # 二级标题(##)细分。这样每块都能正确归属到它所属的主题，不会串台。
        topics = re.split(r"(?=^# )", text, flags=re.MULTILINE)
        for topic in topics:
            topic = topic.strip()
            if not topic:
                continue
            h1 = re.match(r"^#\s+(.+)$", topic, flags=re.MULTILINE)
            topic_name = h1.group(1).strip() if h1 else md.stem
            subparts = re.split(r"(?=^## )", topic, flags=re.MULTILINE)
            for part in subparts:
                part = part.strip()
                if not part:
                    continue
                h2 = re.match(r"^##\s+(.+)$", part, flags=re.MULTILINE)
                section = h2.group(1).strip() if h2 else topic_name
                # 正文里补上主题名，避免"缓冲与同步"这种小节名脱离主题后语义不清
                body = part if h1 else f"# {topic_name}\n{part}"
                chunks.append({
                    "id": hashlib.md5(f"{md.name}:{topic_name}:{section}".encode()).hexdigest()[:12],
                    "source": md.name,
                    "topic": topic_name,
                    "section": section,
                    "text": body,
                })
    return chunks


# ------------------------------------------------------------
# 环节 2：向量化
# 用 Ollama 的 OpenAI 兼容 /embeddings 接口。nomic-embed-text 每次调一个文本，
# 这里逐个调用（文本量小，够用；量大时改成批量）。
# ------------------------------------------------------------
def embed(texts):
    vectors = []
    for t in texts:
        resp = client.embeddings.create(model=EMBED_MODEL, input=t)
        vectors.append(resp.data[0].embedding)
    return vectors


# ------------------------------------------------------------
# 环节 3：建索引并持久化
# 计算文档指纹(corpus 内容哈希)，没变化就复用旧索引，避免每次重算。
# ------------------------------------------------------------
def _corpus_fingerprint(chunks):
    joined = "".join(c["id"] + c["text"] for c in chunks)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def build_index(force=False):
    chunks = load_and_split()
    fp = _corpus_fingerprint(chunks)

    if INDEX_FILE.exists() and not force:
        saved = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        if saved.get("fingerprint") == fp:
            print(f"[索引] 复用现有索引（{len(saved['chunks'])} 个块），指纹一致。")
            return saved

    print(f"[索引] 开始构建：{len(chunks)} 个块，embedding 模型={EMBED_MODEL} ...")
    vecs = embed([c["text"] for c in chunks])
    index = {"fingerprint": fp, "model": EMBED_MODEL, "chunks": chunks, "vectors": vecs}
    INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"[索引] 已写入 {INDEX_FILE.name}")
    return index


# ------------------------------------------------------------
# 环节 4：检索
# 余弦相似度排序，取 top-k。返回带来源的片段文本。
# 注意：这就是 Step2 里 get_go_doc 要返回的"文档"——只不过现在不是字典
# 精确匹配，而是语义相似度匹配，所以问法变了也能命中。
# ------------------------------------------------------------
def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-10)


def rag_search(question, top_k=3, index=None):
    index = index or build_index()
    qvec = embed([question])[0]
    scored = []
    for chunk, vec in zip(index["chunks"], index["vectors"]):
        scored.append((_cosine(qvec, vec), chunk))
    scored.sort(key=lambda x: x[0], reverse=True)

    hits = scored[:top_k]
    print(f"      [RAG检索] top-{top_k} 命中：" +
          ", ".join(f"{c['source']}#{c['section']}({s:.3f})" for s, c in hits))
    # 拼成一段"参考资料"，并附带来源，供模型引用
    body = "\n\n".join(
        f"---\n来源：{c['source']} → {c['section']}\n{c['text']}" for _, c in hits
    )
    return body


if __name__ == "__main__":
    force = "--rebuild" in sys.argv
    idx = build_index(force=force)
    # 取第一个非 -- 开头的参数作为问题，允许 "rag_core.py --rebuild 问题" 这种顺序
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    demo = positional[0] if positional else "channel 是怎么做同步的？"
    print("\n[示例检索] 问题：", demo)
    print("\n[返回给模型的参考资料]\n")
    print(rag_search(demo, index=idx))

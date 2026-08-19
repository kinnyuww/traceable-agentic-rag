# v0.1 索引升级报告：自适应 Chunk 与可选择 Dense Search

> 本报告只回答当前最重要的三个问题：数据如何切、SQLite/向量如何搜、问题如何被
> Agent 理解。Docker/MCP 等已有内容不重复展开。

## 一页结论

这次没有把所有知识库强行迁到 ANN，也没有为每个 chunk 增加一次生成模型调用。
系统新增了两个互相独立、固化在不可变索引版本里的选择轴：

| 选择轴 | 默认 | 可选 | 核心取舍 |
|---|---|---|---|
| Chunk | `auto` | `structure / semantic` | 好结构不破坏；弱结构才付语义分析成本 |
| Dense | `auto` | `exact / hnsw` | 小库保留完整召回；规模上来后用图索引换延迟/内存 |

```text
文档 → parser hard boundary
     → Auto: 标题结构好？structure : semantic breakpoint
     → chunk + source offsets
     → embedding
     → SQLite chunks + FTS5 BM25
     → Exact matrix 或 USearch HNSW

问题 + 可选会话 → DeepSeek query understanding（失败时规则回退）
     → original / canonical / 多跳子问题独立召回
     → 各自 Dense Top 20 + BM25 Top 20 + RRF Top 12
     → 合并去重 → canonical query 全局 rerank Top 6
     → Evidence Set → Gate → 同一集合生成/引用
```

## 1. 修改前后最重要的变化

### 修改前

- Dense：每个查询从 SQLite 读取当前索引全部 embedding BLOB，在 Python 中逐行
  归一化并计算精确余弦。质量是完整召回，但重复 I/O、反序列化和线性计算。
- Chunk：在 parser section 内按约 1100 字符切，优先空行或“标点后空白”。连续
  中文 `。下一句` 因没有空白，经常退化为硬字符截断。
- Query understanding：代码已有 greeting/help、ambiguity、multi-hop 规则，但除
  `question_type` 外没有形成一个独立、容易理解的 trace 事件。

### 修改后

- Exact 仍是金标准，但不可变索引首次加载后缓存为归一化 NumPy 矩阵；不再每问
  一次都解码整库向量。
- HNSW 成为真正可用的本地后端：建索引时生成 `.usearch` 图和 chunk-ID manifest，
  查询时 memory-map；请求值、解析值、参数和延迟可追踪。
- Auto 默认少于 100,000 chunks 使用 Exact，达到阈值才使用 HNSW。阈值可配置。
- Chunk Auto 对有意义的 Markdown/DOCX 标题保留结构路径，对长 TXT/PDF 通用段落
  才运行 embedding 语义断点；中文句界已修正。
- `query_understanding` 现由 DeepSeek 生成 canonical query、single/multi-hop、实体、
  约束和初始子问题；失败时保留确定性规则路径。模型耗时、token、retry 和错误可追踪。

## 2. “SQLite 搜索算法”到底是什么

这个项目不是只有一种 SQL 搜索。在线检索有两个召回通道：

### 2.1 Sparse：SQLite FTS5 BM25

`chunks_fts` 是 FTS5 虚拟表。查询先提取英文/数字 token 和中文字符/双字词，经过
安全转义后用 `OR` 连接，再由 SQLite `bm25()` 返回 Top 20。

注意：FTS5 的 BM25 数值越小通常越好。代码没有把 BM25 原始值与 cosine 直接
相加，而是只取两个列表的名次做 RRF：

```text
RRF(chunk) = Σ 1 / (60 + rank)
```

所以 Dense 与 BM25 不需要做脆弱的分数标定。两边都命中的 chunk 会自然获得更高
RRF 排名；只被其中一边发现的专名或语义近义证据也不会立刻消失。

### 2.2 Dense Exact：不是 SQL 向量扩展

SQLite 保存 float32 BLOB 和元数据，但精确 cosine 由 NumPy 执行：索引版本首次
使用时加载为矩阵、逐行归一化；每次 query embedding 归一化后执行矩阵乘法。

它的优势是相对于库内全部向量 Recall=1，特别适合：

- 小中型本地知识库；
- 评测 HNSW 是否漏召回的 oracle；
- 排查“是 embedding 不好，还是 ANN 没找到”。

它的代价是矩阵内存和 O(N×D) 计算，而不是“搜索能力太强导致后面都没用”。即使
Exact 找到了最相近 Top 20，BM25 仍负责编号/专名，reranker 仍负责 query-aware
排序，Gate 仍负责充分性与多来源检查；各层解决的问题不同。

### 2.3 Dense HNSW：图近邻而不是全扫描

HNSW 把每个向量连到分层近邻图；查询从稀疏高层逐步下沉，在有限候选区域探索，
因此不需要访问每个向量。这里使用 USearch 2.26 的本地持久化实现，适配 macOS
ARM64 和 Linux ARM64，不要求另起 Qdrant/Postgres 服务。

默认参数偏向召回：`M=32, efConstruction=512, efSearch=512`。它们不是事实可信度
参数，只控制图连接、构建搜索宽度和查询搜索宽度。

官方 pgvector 文档同样把 HNSW 与 IVFFlat 的关系概括为：HNSW 通常有更好的
速度-召回权衡，但构建慢、内存更多；IVFFlat 把向量聚类到 lists、查询只 probe
部分 lists，构建快、内存较低，但需要训练数据和更多参数选择。当前本地版优先
HNSW，IVF 留给真正的大规模数据实验，而不是同时塞进 v0.1 增加表面选项。

参考：[pgvector indexing](https://github.com/pgvector/pgvector#indexing)、
[Qdrant indexing](https://qdrant.tech/documentation/manage-data/indexing/)、
[USearch](https://github.com/unum-cloud/USearch)。

## 3. 为什么 Auto 阈值是 10 万，而不是几千

本机固定种子 microbenchmark：20,000 个随机 384 维向量、100 个查询、Top 20。

| 后端 | Recall@20 vs Exact | P50 | P95 | 额外构建 |
|---|---:|---:|---:|---:|
| Exact cached matrix | 1.000 | 1.04 ms | 1.28 ms | 无 |
| HNSW 32/512/512 | 0.966 | 18.31 ms | 19.30 ms | 48.12 s |

结论不是“HNSW 不好”，而是小库里 BLAS 矩阵乘法已经非常快，图遍历反而没有
优势。HNSW 的价值会在矩阵内存、并发、P95 或 N 足够大时出现。因此 Auto 采用
保守的 100,000 起点，并允许部署者修改。完整结果在
[`results/dense-backend-synthetic.json`](results/dense-backend-synthetic.json)。

该数据是随机向量，只验证实现和取舍，不能替代中文/英文领域查询的 Recall@K。
发布 HNSW 索引前仍应在固定 gold query 上比较：

```text
ANN Recall@20 = |HNSW Top20 ∩ Exact Top20| / 20
```

同时看最终 rerank Hit@6；ANN 少一个无关近邻不一定伤害 RAG，漏掉 gold evidence
则必须阻止发布。

## 4. 自适应 Chunk 的实际算法

### 4.1 Parser 结构仍然是硬边界

- Markdown/DOCX 标题形成 section；
- PDF 每页形成 section；
- TXT 默认一个 `Document` section。

Chunk 不跨这些边界。语义策略不是把全文先打散再任意重组。

### 4.2 Auto 如何判断结构强弱

- section 名是具体标题，或文本不超过 target：走 `structure`；
- section 是空、`Document`、`Page N` 且足够长：走 `semantic`；
- 同一文档可以一部分 structure、一部分 semantic，结果记为 `hybrid`。

这是一条可复现的确定性规则，而不是让 LLM 猜“文档质量”。

### 4.3 Semantic 如何找边界

1. 找到中英文句界与空段；
2. 每句组合前后各一句作为 window；
3. 用当前本地 embedding 批量编码；
4. 计算相邻 window 的 cosine dissimilarity；
5. 取第 90 百分位以上作为候选主题转折；
6. 在候选中选择最接近 1100 字符、且位于 320–1600 范围的点；
7. 没有合格语义点时回退句界，再回退硬上限。

这个思路与 [LlamaIndex SemanticSplitter](https://developers.llamaindex.ai/python/framework-api-reference/node_parsers/semantic_splitter/)
一致，但增加 parser 硬边界与 min/max，避免异常文档生成极小或无限大 chunk。

### 4.4 为什么这版不生成 chunk 摘要

[Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
是在每个 chunk 前补充文档级语境，再做 embedding/BM25。它可能提升局部片段的
可检索性，但如果摘要被 Gate 当证据，模型生成的误读会污染证据链。

所以当前版本只用 embedding 决定边界，不改写原文。`contextual_text` 仍只有
Document/Section/Page 元数据；生成式 contextual summary 保留为后续独立 A/B：

- 只能作为 retrieval hint，不能作为可引用事实；
- Gate coverage 应只计算原始 `text`；
- 必须记录生成模型、prompt、成本和摘要-原文冲突率；
- 没有显著 Recall/Hit@6 增益就不应成为默认。

## 5. Query understanding 在当前 Agent 中怎样体现

当前先保留 greeting/help 的确定性无 RAG 路由。普通知识问题会把原问题、知识库
简介和同一 `conversation_id` 下最近 0～10 轮发送给 DeepSeek；模型只做理解和规划，
不回答问题。它返回 standalone canonical query、single/multi-hop、实体、约束、
clarify 状态和多跳 2～3 个原子子问题。

Single-hop 在原问题与 canonical 不同时分别召回；Multi-hop 使用 canonical 加原子
子问题，总视角最多四个。每个视角独立执行 Dense/BM25/RRF，候选按 chunk ID 合并，
最后只针对 canonical query 做一次全局 rerank Top 6。EvidenceSet 使用这一次分类，
不会在 rerank 后再次用 `_looks_multihop` 覆盖它。DeepSeek 禁用、失败或返回非法类型
时才回退到确定性规则。

`query_understanding` trace 记录是否实际尝试 LLM、method、canonical query、分类、
会话轮数、子查询、latency、token、retry 和 model error；`retrieval_merge_rerank`
记录并发、合并去重和全局 Top 6。完整复核与同环境 A/B 见
[`CHUNK_AND_QUERY_UNDERSTANDING_REVIEW.md`](CHUNK_AND_QUERY_UNDERSTANDING_REVIEW.md)。

## 6. 用户怎样选择

Web UI 的“文档与索引”页有两个下拉框。REST 请求相同：

```json
POST /v1/knowledge-bases/{kb_id}/index-builds
{
  "activate": true,
  "chunk_strategy": "auto",
  "dense_backend": "auto"
}
```

推荐起点：

| 场景 | Chunk | Dense |
|---|---|---|
| 标题清晰的 Markdown/DOCX，小库 | `auto` 或 `structure` | `exact`/`auto` |
| 长 TXT、普通 PDF、结构弱 | `auto` | `exact`/`auto` |
| 边界实验 | `semantic` 与 `structure` 建两个 index | 固定 `exact`，避免混入 ANN 变量 |
| 大库延迟/内存实验 | 固定 chunk 策略 | `hnsw` 对 `exact` 做 recall 对照 |

不要同时更换 chunk、embedding、HNSW 和 reranker 后再比较结果，否则无法知道收益
来自哪里。不可变 index version 和显式配置正是为了单变量实验。

## 7. 涉及文件与验证

| 文件 | 变化 |
|---|---|
| `src/ragagent/dense_index.py` | Exact cache、USearch HNSW 构建/持久化/查询 |
| `src/ragagent/retrieval.py` | 按 index config 路由 Dense，并写 backend/latency trace |
| `src/ragagent/chunking.py` | 中文句界、自适应 strategy、语义 sentence-window 断点 |
| `src/ragagent/ingestion.py` | 固化策略、语义 chunk、HNSW artifact、修复跨版本 chunk ID 冲突 |
| `src/ragagent/agent.py` | 显式 `query_understanding` trace |
| `src/ragagent/schemas.py` / `config.py` | REST 选择与部署参数 |
| `src/ragagent/static/*` | Web UI 策略选择 |
| `tests/*` | 结构/语义路径、HNSW/Exact 对照、trace/API/UI 契约 |

当前验证：Ruff 全通过，完整 pytest `46 passed`。新增 A/B 测试还暴露并修复了一个
旧 bug：同一知识库第二次建索引时，旧 chunk ID 未包含 index version，会违反全局
唯一键。现在 chunk ID 绑定不可变 index version，才能真正建立两套策略做回放。

Docker/真实模型黑盒也完成：强制 `semantic + hnsw` 对本项目 README 建索引，
Qwen3-Embedding 0.6B 实际编码 6 个语义句窗、选出 1 个断点，最终得到 17 个
1024 维 chunks 和 HNSW artifact；建索引 3.417 秒。DeepSeek V4 Flash 对“Auto
何时切 HNSW”给出正确的 10 万阈值答案，`query_understanding`、HNSW backend、
Gate/生成/引用 ID 一致性均出现在 trace。

该真实运行还捕获一次独立的 reranker HTTP 500：7.649 秒后降级 lexical rerank，
最终回答仍正确但总耗时 13.190 秒。这不是 Chunk/HNSW 失败，却是下一阶段值得单独
修复的可靠性与超时问题；本次没有用扩大范围的方式掩盖它。机器可读摘要在
[`results/strategy-smoke-real-model.json`](results/strategy-smoke-real-model.json)。

## 8. 下一步真正值得做的 hooks

1. **领域 gold evidence A/B**：固定文档和 query，比较 structure/semantic 的
   chunk evidence recall、Hit@20、rerank Hit@6 与重复率。
2. **Auto 不只看数量**：以可用内存、向量维数、并发和实测 P95 决定 Exact/HNSW，
   而不只是 100k 静态阈值。
3. **父子 chunk**：小 chunk 用于召回，命中后扩展到父 section；扩展后的最终集合
   必须在 Gate 前确定，继续保持 Gate/生成一致。
4. **生成式 Contextual Retrieval 实验**：明确 retrieval hint 与原文 evidence 的
   数据类型边界，防止摘要污染 Gate。
5. **更好的 parser**：DOCX 表格、PDF layout/OCR 往往比继续调 chunk 更先决定
   “证据是否进入系统”。

这版的核心不是宣称找到唯一最优算法，而是让不同规模、不同结构的知识库可以选择，
并且每个选择都可冻结、可对照、可追踪。

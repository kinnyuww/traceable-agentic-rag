# Traceable Agentic RAG v0.2.0：完整实现、评测与路线图报告

发布日期：2026-08-20
面向对象：希望直接使用系统的人、准备把它接入自己 Agent 的开发者，以及后续负责
Harness Engineering / HITL 的工程人员。

## 1. 先说结论

当前项目已经是一个可运行的端到端 Agentic RAG 应用和服务，而不再只是若干 RAG
算法的拼接示例。用户可以上传真实文档、建立不可变索引、在网页问答、检查引用和
每一步 trace；外部 Agent 也可以通过同一 REST API 建库、检索或让本系统完成证据
审核与最终回答。

本次发布建议定为 **v0.2.0**，而不是 v0.1.x，原因是在线检索的语义已经发生了明确
扩展：系统加入了 Query Understanding、会话指代消解、多视角召回和统一全局 rerank，
离线侧也从固定结构切块与全量精确 Dense Search 扩展到了自适应切块和可选 HNSW。

它现在可以被准确描述为：

> 一个 query-aware、bounded、traceable、evaluable 的本地 Agentic RAG 应用/服务。

它现在还不能被准确描述为“会自动自我进化的 RAG”。反馈采集、自动失败归因、候选
参数实验、回归门禁、HITL 审批和配置发布/回滚尚未实现。这些是 v0.3 的工作，而
v0.2 提供了做这些工作的 trace、金标评测和不可变版本基础。

## 2. 三种颗粒度的心智模型

### 2.1 一句话模型

文档先被加工成可引用证据；问题先被规范化，再从多个角度寻找证据；系统只在证据
门判断充分时，才把同一批证据原样交给 DeepSeek 回答。

### 2.2 产品模型

```text
Web 用户 ─┐
外部 Agent ├─→ REST / Web Application ─→ 同一个 RAG Core
自动脚本 ─┘                            ├─→ SQLite + 原始对象 + 向量索引
                                      ├─→ 本地 Qwen embedding / reranker
                                      ├─→ DeepSeek v4 flash
                                      └─→ 不可变 run trace 与 evaluation
```

Web UI 不是旁路 demo，REST 也不是另一套实现。两者操作相同知识库、相同索引版本、
相同 Agent 状态机和相同 trace。因此目前支持三条实际路线：

1. 人在 Web UI 上传、建库并问答；
2. 人在 Web UI 维护知识库，外部 Agent 调用 `/v1/query`；
3. 程序完全通过 REST 创建知识库、上传、轮询、建索引并调用。

### 2.3 工程数据流模型

离线建库：

```text
Upload
  → 安全文件名、大小限制、SHA-256 内容寻址保存
  → PDF / DOCX / Markdown / TXT 解析
  → parser 硬边界 + 自适应 Chunk
  → Qwen 1024 维 embedding
  → SQLite FTS5 BM25 索引
  → Exact 矩阵或 USearch HNSW 图
  → 新的 immutable index_version
  → 全部成功后原子激活
```

在线问答：

```text
Question + optional conversation_id
  → DeepSeek Query Understanding（失败则确定性回退）
  → canonical query + single/multi + 检索视角
  → 各视角并发执行 Dense Top20 + BM25 Top20 + RRF Top12
  → 按 chunk_id 合并去重
  → 只用 canonical query 做一次全局 Rerank Top6
  → single: 动态 floor 后最多 4；multi: Top6 全保留
  → Evidence Gate 审核这一最终集合
  ├─ answer: 同一集合直接生成并引用
  ├─ retry: 最多 4 个补充 query，再检索一次并重新过 Gate
  └─ clarify / 第二轮仍不足: 澄清或拒答
```

整个 Agent 最多两轮，没有无界循环，也不会在在线请求中自行修改 chunk 参数、模型或
阈值。这种“有预算的 Agentic”比开放式循环更容易复现、评测和后续做受控优化。

## 3. 从 Chunk 到完整 Agentic RAG，分别做了什么

### 3.1 文档与 Chunk

当前 Chunk 策略为 `auto / structure / semantic`：

- `structure` 尊重 Markdown/DOCX 标题、PDF 页等 parser 硬边界，再按句界和长度确定性
  打包；
- `semantic` 在单个 parser section 内切分句子，构造带一个相邻句缓冲的窗口，用本地
  Qwen embedding 计算相邻窗口的余弦差异，并在第 90 百分位差异处形成主题断点；
- `auto` 对有意义标题的结构化 section 使用 structure，对长 `Document` 或 `Page N`
  这类弱结构 section 使用 semantic；
- 目标长度约 1100 字符，最大约 1600，重叠约 160；中文 `。！？` 可直接识别；
- 文件、页/章节、字符 offset 和原始文本始终保留，生成模型不会重写原文证据。

这里的“语义切块”是 **本地 embedding 做边界检测**，不是让 DeepSeek 生成摘要。当前
embedding 模型是 `ai/qwen3-embedding:0.6B-F16`，1024 维，经 Docker Model Runner
的 llama.cpp/Metal 路径运行。`contextual_text` 只包含确定性的文件/章节元数据；
Anthropic-style Contextual Retrieval 暂未作为默认路径，以免模型生成的上下文被误当
成原始事实。

这套策略的现实取舍是：弱结构资料有机会得到更合理的边界，但索引时间会增加，而且
更漂亮的 chunk 边界并不自动等于更好的最终回答。v0.2 保持 `auto` 默认，并把
requested/resolved strategy、语义窗口数、断点数和耗时写入索引配置或 trace。

### 3.2 “SQL 搜索”实际由什么组成

项目没有让 SQLite 承担所有向量算法。当前是两个正交通道：

1. **稀疏检索**：SQLite FTS5 的 BM25。为了兼顾中英文，会写入 Latin token、中文
   单字与 bigram；它擅长编号、名称、原词和关键词命中。
2. **Dense 检索**：向量和 chunk 元数据持久化在 SQLite，但 Exact 查询会把归一化
   向量缓存成 NumPy 矩阵并做精确 cosine；HNSW 则使用持久化的 USearch sidecar 图。

每个不可变索引可以选：

- `exact`：不近似，适合小库、基准评测和要求精确复现的场景；
- `hnsw`：近似最近邻，适合大库和低延迟，但必须测 Recall@K；
- `auto`：当前保守地在 100,000 chunk 以下使用 Exact，达到阈值才切 HNSW。

Dense 和 BM25 各取 Top 20，RRF 以 `k=60` 按排名融合到 Top 12，然后 Qwen reranker
输出 Top 6。HNSW 只是 Dense 的候选生成方式，不会替代 BM25、RRF、rerank、Gate
和第二轮规划。

### 3.3 Query Understanding 怎样体现“Agentic”

普通知识问题不再直接拿原问题只检索一次。DeepSeek 官方 `deepseek-v4-flash` 首先
接收原问题、知识库名称/描述和同一 `conversation_id` 下最近 0～10 轮历史；它只做
理解，不允许回答，并返回：

- 无指代、保留名称/数字/年份/否定和范围的 `canonical_query`；
- `single-hop` 或 `multi-hop`；
- intent、entities、constraints；
- 必要时的 clarification；
- 多跳问题的 2～3 个原子检索子问题。

单跳通常以 `[original, canonical]` 两个去重视角召回；多跳以
`[canonical, atomic subqueries...]` 召回，最多四个视角，默认并发二。各视角只负责
召回，候选合并后统一用 canonical query rerank 一次。这样既利用了不同检索表述，
又避免不同 query 的 rerank 分数无法直接比较。

DeepSeek 不可用时，系统回退到 `_looks_multihop` 等确定性规则并把错误、重试和回退
方法写入 trace。会话历史只进入 Query Understanding，不被当作事实证据直接发给最终
生成模型。

### 3.4 Evidence Gate 与最终上下文

这是 v0.2 最重要的正确性约束：

```text
Rerank Top6
  → 构建唯一 Evidence Set
  → Gate 审核
  → 原样生成
  → 从同一集合构建 citation
```

单跳使用 `score_floor = max(0.02, top1_rerank × 10%)` 降噪并最多保留四块；若全部
低于门槛则保留 Top 1。多跳最多使用六块，而且不使用 score floor 删除后排证据，
避免排名第 5/6 的另一来源刚好支撑第二个事实却被提前丢弃。

Gate 使用 Top 1 rerank 作为相关性主信号，并以所有已选 chunk 合并后的联合 query
coverage 作为覆盖信号；多跳来源多样性和灰区 DeepSeek 证据充分性分类都能看到完整
Top 6。rerank rank/score 会作为“相关性提示”传入最终提示词，但明确不代表事实正确、
来源权威或信息最新。

Gate 放行后没有第二套 Context Selection。trace 中的
`evidence_gate.audited_chunk_ids`、`answer_generation.selected_chunks`、
`answer_generation.citation_chunk_ids` 以及 API 返回 citation 的 chunk IDs 可直接核对。
这消除了旧设计中“Gate 因某个 chunk 放行，但生成前又把这个 chunk 删除”的状态漂移。

### 3.5 Trace 与评测

离线阶段保存 upload、parse、chunk、embedding、persist、activate 等 job event；在线
阶段保存：

- 原问题、canonical query、会话历史条数、问题类型和理解模型用量；
- 每个 query view 的 Dense/BM25/RRF 候选、分数、来源和时延；
- 合并前后数量、去重数、唯一全局 rerank Top 6；
- Evidence Set 的 limit、floor、selected/discarded IDs；
- Gate 实际审核 IDs、联合覆盖、来源多样性、决策和缺失事实；
- 第二轮 query plan、停止原因；
- 最终生成 IDs、citation IDs、模型用量、降级和总时延。

系统因此能区分“答案碰巧正确”和“链路干净”：解析漏了、chunk 切坏、Dense/BM25
没召回、RRF/rerank 降权、Gate 误拒、生成忽略证据或引用映射错误，都有不同证据。

## 4. v0.1 到 v0.2 的实际变化

| 维度 | v0.1 baseline | v0.2.0 |
|---|---|---|
| Chunk | 结构优先、固定边界 | structure/semantic/auto，自适应弱结构语义断点 |
| Dense | 小库精确 cosine 基线 | exact/hnsw/auto，缓存矩阵与持久化 ANN 图 |
| 问题理解 | 确定性单/多跳判断 | DeepSeek canonical query + 约束/实体/澄清 + 确定性回退 |
| 会话 | 每次请求无状态 | 可选 conversation_id，0～10 轮隔离记忆 |
| 首轮召回 | 一个 query | 最多四个有界视角并发召回 |
| Rerank | 每次检索各自 rerank | 合并去重后对 canonical query 做一次全局 rerank |
| Evidence | Gate 与后续上下文曾可能分离 | 一个 Evidence Set 贯穿 Gate、生成与引用 |
| 多跳尾部证据 | 可能被动态 floor 删除 | Top 6 全保留，并检查全部来源 |
| 可视化 | 通用 trace 页面 | 增加完整在线检索字段流转图 |
| 自调优 | 未实现 | 仍未实现，但已有更完整的观测和 A/B 基础 |

## 5. 验证结果与应该怎样解读

### 5.1 工程回归

- Ruff：通过；
- pytest：46 个测试通过；
- Docker Compose 配置：通过；
- Docker API 与 worker：健康；
- DeepSeek 官方 `deepseek-v4-flash`：实际 Query Understanding 与生成调用通过；
- Gate 审核、生成和 citation ID 一致性：单测和真实请求均通过。

### 5.2 同运行时新旧 Query Understanding 对比

| 数据集/指标 | 旧路径 | v0.2 路径 | 观察 |
|---|---:|---:|---|
| QASPER 完整 gold 证据进入 rerank 集合 | 91.67% | 91.67% | 持平 |
| QASPER 平均延迟 | 2591.6 ms | 2744.2 ms | +5.9% |
| MultiHopRAG 完整 gold 证据进入 rerank 集合 | 94.44% | 100.00% | +5.56 pp |
| MultiHopRAG 全部 gold 证据被引用 | 94.44% | 100.00% | +5.56 pp |
| MultiHopRAG answerability accuracy | 95.83% | 100.00% | +4.17 pp |
| MultiHopRAG 平均延迟 | 3487.3 ms | 4650.4 ms | +33.4% |

结论不是“所有数据都显著变好”，而是多跳完整证据覆盖得到可见提升，代价是多视角
召回与问题理解增加延迟。单跳 QASPER 基本保持质量，平均延迟小幅增加。

### 5.3 Auto 与强制 Semantic Chunk 对比

在同一 QASPER 样本上：

| 指标 | Auto | 强制 Semantic |
|---|---:|---:|
| 索引耗时 | 19.71 s | 66.84 s |
| chunk 数 | 164 | 167 |
| 完整 gold 证据进入 rerank 集合 | 91.67% | 91.67% |
| 最终 evidence hit（answerable） | 83.33% | 91.67% |
| answerability accuracy | 71.43% | 64.29% |

强制 Semantic 的建索引时间为 Auto 的 3.39 倍，Evidence 命中提高但 Gate 误拒使端到端
answerability 下降。因此 v0.2 保留 Auto 默认，不能把“所有文档都做语义切块”当成
无条件最佳实践。

### 5.4 真实 DeepSeek / Ragas 检查

一次带会话指代的真实请求成功把问题规范化为“Auto Dense 默认在多少个 chunk 后切换
到 HNSW？”，答案为 100,000 chunk，Gate/生成/citation IDs 一致。Query Understanding
耗时约 3.29 秒，总耗时约 21.06 秒。

两条真实样本的当前 Ragas/确定性检查中：gold evidence hit 和全部预期引用均为 1.0，
context precision/recall 均为 0.75。faithfulness 只有一个样本得到 0.667，另一个 judge
因 `max_tokens` 输出不完整失败，因此不能把该单值当作稳定版本结论。

机器可读结果见：

- `reports/results/public-benchmark-f2134c6-same-runtime.json`
- `reports/results/public-benchmark-current-auto.json`
- `reports/results/public-benchmark-current-semantic.json`
- `reports/results/chunk-query-comparison-summary.json`
- `reports/results/live-ragas-current-query-understanding.json`

## 6. 当前最重要的不稳定面

### P0/P1：本地 reranker 长输入失败与延迟

真实请求中，当前 Docker Model Runner 下的 Qwen reranker 对较长 pair 输入可能在
7.6～9.15 秒后返回 HTTP 500，日志指向物理 batch/context 限制。系统按设计降级到
lexical reranker，所以请求仍能正确完成且 trace 可见，但延迟和排序质量会受影响。

下一步应优先选择下面一种可验证方案：

1. 调整 runner 的 batch/context 参数并做长输入回归；
2. 换用明确支持更长 pair 输入的独立 rerank 服务；
3. 做 query-aware evidence condensation，但必须验证 gold span 不会被截断。

不建议无记录地从尾部硬截断，因为关键多跳证据可能就在后半段。

### 其它边界

- PDF 复杂布局、扫描件 OCR、DOCX 复杂表格和图片没有稳定解析；
- Auto 的 100,000 chunk HNSW 阈值只是保守起点，必须在真实领域测 Recall@20、P95、
  内存和并发；
- DeepSeek 启用时，选中的证据 chunk 会离开本机；部署者必须确认数据出境策略；
- 当前是单工作区开发者应用，没有认证、多租户、RBAC、rate limit 和公网安全层；
- Ragas 的 LLM judge 可能受输出长度与模型波动影响，不能替代确定性证据指标和专家金标。

## 7. 对照最初计划：现在到哪一步

| 里程碑 | 目标 | 状态 |
|---|---|---|
| v0.1 Traceable Baseline | 端到端建库/问答、Hybrid RAG、Gate、引用、trace、评测 | 已完成并发布 |
| **v0.2 Query-Aware Adaptive Retrieval** | 自适应 chunk/ANN、问题理解、会话记忆、多视角召回、证据集合一致性 | **当前完成** |
| v0.3 Harness Engineering + HITL | 反馈金标、自动失败归因、候选实验、sealed regression、审批发布/回滚 | 下一阶段 |
| v0.4+ Production & Protocols | MCP、OCR/表格、多租户安全、漂移监控、受限自动提升 | 后续规划 |

如果把最初愿景拆成“可用 RAG Agent、可观察、可评测、可诊断、可受控优化、可自主
提升”六级，当前已经完成前三项并为第四项准备了完整数据；第五项是 v0.3 的核心，
第六项只有在 HITL 与回归门禁可靠后才应该逐步开放。

## 8. v0.3 建议范围

v0.3 不应直接做“线上自动调参”，而应先做可回滚的 Harness：

```text
用户/专家反馈
  → 固化问题、期望答案、gold evidence 与不可回答标签
  → 比较每层候选列表并归因
  → 只生成单变量候选实验
  → development set A/B
  → sealed regression set
  → 质量 / 成本 / P95 门禁
  → HITL 审批
  → 发布新的配置或索引版本
  → 可回滚
```

建议 v0.3 的最小交付包括：

1. Web/API feedback schema：正确/错误、缺失事实、gold chunk、期望拒答；
2. 失败分类器：parse/chunk/dense/sparse/fusion/rerank/gate/generation/citation；
3. 单变量实验描述与可复现 dataset snapshot；
4. development 与 sealed test 分离，防止对测试集调参；
5. 质量、延迟、成本三类门禁；
6. 专家批准、配置版本发布和一键回滚；
7. 先修复 reranker 长输入，再把 chunk/Gate 候选加入自动实验空间。

MCP 可以与 v0.3 并行或排在其后。现有 REST schema 已足够稳定，MCP 应只是把
`query`、`retrieve`、`upload_document`、`get_run` 等能力映射为 tools/resources，
不应复制一套检索核心。

## 9. 最终发布判断

v0.2.0 适合发布到 GitHub main，理由是：

- 从文档输入到引用回答的主链路完整；
- Agentic 行为有明确预算与停止条件；
- 新旧数据流、Evidence Set 边界和失败降级都有自动测试；
- 真实模型、公开样本和粗糙知识库都已经用于验证；
- 已知问题没有被隐藏，reranker 降级、Ragas judge 错误和 Semantic/Gate 负向波动都被
  明确记录；
- README、架构、运维、使用模式、失败模型和机器可读结果可以支持别人复现。

但它仍应保持 0.x：自调优闭环、复杂文档解析和生产安全层尚未完成。当前最合适的
对外定位不是“全自动自进化平台”，而是“为 Harness Engineering 准备好的、可运行且
可审计的 Agentic RAG baseline”。

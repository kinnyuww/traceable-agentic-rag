# Chunk 与 Query Understanding 综合复核

日期：2026-08-19
结论版本：当前工作区（侧边会话 Query Understanding 改动优先）

## 先看结论

当前 Chunk 方案不是“让大模型逐块总结”，而是一个离线、可复现的自适应边界算法：

```text
文档解析出的硬边界
  → Auto 判断该 section 的结构是否可信
  → 结构好：按自然句界切
  → 结构弱且较长：Qwen embedding 找主题转折点
  → 约 1100 字符、160 字符重叠、最多 1600 字符
  → 原文 chunk + 文件/标题/页码/offset
  → 再做正式索引 embedding、BM25 和 Dense 索引
```

本轮复核后的判断：

- `auto` 继续作为默认值是合理的。它对结构清晰的 Markdown/DOCX 不增加语义切块
  成本，对长 TXT、PDF 页和通用 `Document` section 才启用语义断点。
- Semantic 使用的是当前索引配置中的同一个本地 embedding 模型。现在真实部署为
  `ai/qwen3-embedding:0.6B-F16`（1024 维，Docker Model Runner / llama.cpp / Metal），
  不是 DeepSeek，也没有额外的在线生成调用。
- 强制 `semantic` 不是普遍更好。在 5 篇 QASPER 论文上，它让建索引耗时从
  19.71 秒增加到 66.84 秒（3.39 倍），Rerank 的文档级完整召回没有提升；它修复了
  1 个最终证据命中，但该题随后仍被 Gate 拒答，说明瓶颈已经转移到 Gate 校准。
- 侧边版 Query Understanding 与 Chunk 没有数据流冲突：Chunk 是离线索引策略；
  Query Understanding 是在线 query 规划。它确实改变了先前“只用确定性
  `_looks_multihop`”的设计，现在以 DeepSeek 的分类为主、规则为失败回退。按本次要求，
  当前侧边版是权威版本。
- 在线 EvidenceSet 仍保持：single-hop floor 后最多 4，multi-hop 保留 Top 6；
  Gate、生成和 citation 仍使用完全相同的 chunk 集合。

## 1. 修改前后的 Chunk 到底差在哪里

### 旧版 `structure/sentence` baseline

旧版已经保留 parser section，但 section 太长时主要做以下处理：

1. 目标长度约 1100 字符；
2. 优先找空行，或“标点后有空白”的句界；
3. 找不到就按字符硬切；
4. 下一块从前一块末尾向前约 160 字符开始。

主要问题：

- 连续中文常见写法 `第一句。第二句。` 的句号后没有空格，旧正则可能识别不到；
- 它知道标题/页码，却不知道同一长 section 内何处发生主题转换；
- overlap 可能从普通空白开始，而不是从完整句子开始；
- 所有文档只有一套策略，无法对结构好和结构弱的资料区别处理。

### 当前 `adaptive_v2`

当前索引可选择：

| 策略 | 实际行为 | 推荐场景 |
|---|---|---|
| `structure` | parser section 内按多语言句界/段落确定性切分 | 标题可靠的 Markdown、DOCX；回放基线 |
| `semantic` | 所有超过 target 的 section 都分析语义转折 | 单变量实验；主题密集的长段落 |
| `auto` | 有意义标题走 structure；通用长 section 走 semantic | 默认混合语料 |

三个策略都遵守同一组不可变约束：

- parser section 永远是硬边界，不跨标题、DOCX Heading 或 PDF 页拼接；
- 小于等于 1100 字符的 section 原样保留，不为了“显得智能”强行切碎；
- 默认 target 1100、overlap 160；
- semantic 的可选边界范围为 320～1600 字符；
- chunk 保存 document ID、filename、section、page、ordinal 和绝对字符 offset；
- `contextual_text` 只有确定性元数据，不含模型生成的事实摘要。

## 2. 每种文件先怎样形成硬边界

Chunk 之前先由 parser 形成 `ParsedSection`：

| 输入 | parser 硬边界 |
|---|---|
| Markdown | `#`～`######` 标题 |
| DOCX | Word 的 Heading 样式 |
| PDF | 每一页 |
| TXT | 整个文件是一个名为 `Document` 的 section |

因此 Semantic 不是全文任意聚类。它只能在一个 parser section 内选择切点。例如 PDF
第 3 页末尾和第 4 页开头即使语义连续，本版也不会合并。这保证引用页码和 offset
稳定，但跨页表格/段落仍是已知局限。

## 3. Auto 怎样判断结构好不好

对每个 section 独立判断：

```text
section 长度 ≤ 1100
  → structure，不切

请求 strategy=structure
  → structure

请求 strategy=semantic 且 section 较长
  → semantic

请求 strategy=auto 且 section 名为空 / Document / Page N
  → semantic

其他有具体标题的长 section
  → structure
```

同一篇文档可能同时包含两条路径，此时索引诊断写作 `hybrid`。这里的“结构判断”是
确定性规则，不调用 DeepSeek，也不会因重复运行而变化。

当前规则仍比较朴素：一个内容混乱但恰好有具体标题的长 section 会被视作结构良好。
下一阶段若要优化，应先在失败集上增加“标题密度、段落长度分布、列表/表格比例”等
可解释特征，而不是直接让另一个 LLM 判文档质量。

## 4. Semantic 的算法与模型

### 4.1 句界

先识别：

- 中文 `。！？`，允许后面没有空格；
- 英文 `.` 只在后面是空白或文本结束时断开，避免把 `3.14` 当两句；
- `!?`、中文引号/书名号后的结束位置；
- 空段落。

### 4.2 带缓冲的 sentence window

若句子为 `S1, S2, S3, ...`，第 i 个 window 是：

```text
Wi = S(i-1) + Si + S(i+1)
```

首尾按实际存在的句子截断。前后各带一句可以降低单句过短时 embedding 抖动，也让
边界判断观察到局部上下文。

### 4.3 Embedding 与距离

所有 window 每 32 个一批送入 `EmbeddingClient`。真实本地配置是：

```text
ai/qwen3-embedding:0.6B-F16
1024 dimensions
Docker Model Runner → llama.cpp → Apple Metal
```

随后对向量 L2 归一化，计算相邻 window 的余弦不相似度：

```text
distance(i) = 1 - cosine(Wi, W(i+1))
```

把本 section 距离分布的第 90 百分位及以上位置标为“候选主题转折”。这不是让模型
生成主题名，而只是用 embedding 判断相邻局部语境变化有多大。

### 4.4 最终切点选择

从当前 chunk 起点出发：

1. 只看 320～1600 字符范围内的语义候选；
2. 选择距离目标 1100 字符最近的候选；
3. 没有合格候选时，回退到最合适的自然句界；
4. 再没有就使用 1600 字符硬上限；
5. 下一块向前保留约 160 字符，并尽量对齐到句首。

如果少于 3 句、所有相邻距离近似相同，或没有有效候选，会安全回退，不会生成空块
或无界大块。

### 4.5 为什么切块阶段看起来会调用 embedding 两次

对走 semantic 的长 section：

1. 第一次 embedding 是 sentence windows，只用来选边界；
2. chunk 确定后，第二次 embedding 是最终 chunk 向量，用于 Dense 检索。

结构路径只有第 2 次。这正是强制 semantic 建索引更慢的原因。两次都复用同一
embedding 服务，不需要新增模型常驻内存。

开发/测试若配置 `embedding_provider=deterministic`，语义切块会使用 feature-hash
向量，只适合确定性测试，不应把该结果当生产语义质量。

## 5. 它不是 Anthropic Contextual Retrieval

当前 `contextual_text` 是：

```text
Document: handbook.md | Section: Refund policy | Page: 4
```

它参与 embedding、BM25 和 rerank，但不改变原文。`contextualize=true` 仍明确失败，
因为本版没有为每个 chunk 生成“这段在整篇文档中的含义”摘要。

这是刻意的证据边界：如果把生成摘要当作 Gate 证据，摘要的遗漏或误读会污染引用链。
以后做 Contextual Retrieval 时，应把生成上下文标为 retrieval hint，Gate 和 citation
仍以原始文本为准，并单独记录模型、prompt、成本和摘要冲突率。

## 6. 侧边版 Query Understanding 的当前数据流

当前在线路径为：

```text
原问题 + KB 简介 + 可选最近 0～10 轮会话
  → DeepSeek Query Understanding（不回答问题）
      ├─ canonical standalone query
      ├─ single-hop / multi-hop
      ├─ entity / constraint / intent
      ├─ clarify 或 retrieve
      └─ 多跳 2～3 个原子子问题
  → 多视角独立 Dense Top20 + BM25 Top20 + RRF Top12
  → 按 chunk ID 合并去重
  → 只用 canonical query 做一次全局 rerank Top6
  → single: floor 后最多4 / multi: Top6 全保留
  → Evidence Gate
  → 同一 EvidenceSet 原样生成与引用
```

Single-hop 在 original 与 canonical 不同的时候保留两个召回视角；相同时自动去重为
一个。Multi-hop 使用 canonical 加 2～3 个原子子问题，总视角最多 4。视角以默认并发
2（请求可设 1～4）运行，但不把不同 query 拼成一条长 query，也不对每个视角分别
rerank；合并后只做一次全局 rerank，使所有候选处在同一分数尺度。

### 与之前设计是否冲突

有一项明确的设计变更，但代码中没有双重判断冲突：

- 之前：`_looks_multihop(question)` 是唯一分类器，不增加 LLM 调用；
- 当前侧边版：DeepSeek 分类是主路径，`_looks_multihop` 只在 LLM 禁用、失败或返回
  非法类型时回退；`understanding.question_type` 是 EvidenceSet 的唯一输入。

因此不会出现“LLM 判 single，rerank 后规则又判 multi”的情况。代价是每个普通知识
问题新增一次外部调用及其成本/抖动。当前 trace 已补充：是否实际尝试 LLM、method、
latency、prompt/completion tokens、retry、model_error 和 fallback method。

这项变更不影响 Chunk：旧索引不需要因为 Query Understanding 升级而重建。

## 7. 公平性能对比

### 7.1 方法

- 机器：M1 Pro 32 GB；评测程序在同一 Linux ARM64 一次性容器中运行；
- Embedding：Qwen3-Embedding 0.6B F16；
- Reranker：Qwen3-Reranker 0.6B；
- Dense：小库 Exact；生成模型关闭，避免生成随机性；
- 固定数据：QASPER 5 文档/14 题、MultiHopRAG 82 文档/24 题、双语控制 6 文档/5 题；
- “修改前”使用提交 `f2134c6`；“当前”使用同一 chunk `auto` 加侧边版多视角架构；
- 这是固定切片诊断，不是数据集官方 leaderboard。

### 7.2 Query Understanding / 多视角：修改前 vs 当前 Auto

| 数据集 | 指标 | 修改前 | 当前 | 变化 |
|---|---:|---:|---:|---:|
| QASPER | Rerank all-evidence@10 | 91.67% | 91.67% | 0 |
| QASPER | 全部 gold 证据被引用 | 83.33% | 83.33% | 0 |
| QASPER | 平均延迟 | 2591.6 ms | 2744.2 ms | +5.9% |
| QASPER | P95 | 4583.9 ms | 5661.2 ms | +23.5% |
| MultiHopRAG | Rerank all-evidence@10 | 94.44% | 100.00% | +5.56 pp |
| MultiHopRAG | 全部 gold 证据被引用 | 94.44% | 100.00% | +5.56 pp |
| MultiHopRAG | answerability accuracy | 95.83% | 100.00% | +4.17 pp |
| MultiHopRAG | 平均延迟 | 3487.3 ms | 4650.4 ms | +33.4% |
| MultiHopRAG | P95 | 8031.6 ms | 9850.5 ms | +22.6% |
| 双语控制 | Rerank/all evidence/answerability | 100% | 100% | 0 |
| 双语控制 | 平均延迟 | 595.9 ms | 615.2 ms | +3.2% |

解释：多视角确实修复了这个固定 MultiHopRAG 切片中的 1 个“完整证据未进入最终
Top6/引用”的问题，但平均延迟增加约三分之一。QASPER 和简单双语题没有质量收益，
只有小幅或噪声级成本。因此当前“single 相同 query 自动去重、multi 才展开”方向
合理，但下一步应评测是否能让高置信 single-hop 跳过 DeepSeek，而不是默认给所有
问题付费。这个优化需作为显式 A/B，不能在本轮偷偷改路由。

注意：上述对照关闭了 DeepSeek，测的是确定性 fallback 下的多视角检索成本；真实
启用 DeepSeek 时还要额外加 Query Understanding 调用。现在已有独立 latency/token
trace，下一批线上回放才能给出稳定分位数。

### 7.3 Chunk：当前 Auto vs 强制 Semantic

QASPER 的 Markdown 标题清晰，因此 Auto 全走 structure；强制 Semantic 才会对长
章节额外分析。

| QASPER 指标 | Auto | 强制 Semantic | 判断 |
|---|---:|---:|---|
| chunks | 164 | 167 | +3 |
| semantic windows | 0 | 756 | 明显增加离线调用 |
| semantic breakpoints | 0 | 87 | 实际参与边界选择 |
| 建索引耗时 | 19.71 s | 66.84 s | 3.39 倍 |
| Rerank all-evidence@10 | 91.67% | 91.67% | 无提升 |
| Rerank Hit@1 | 75.00% | 83.33% | +8.33 pp |
| 最终 evidence hit（answerable） | 83.33% | 91.67% | +8.34 pp |
| answerability accuracy | 71.43% | 64.29% | -7.14 pp |
| 在线平均延迟 | 2744.2 ms | 2727.0 ms | 基本相同 |

变化集中在一个问题 `what language pairs are explored?`：Semantic 把正确论文排到
首位并纳入引用，但 Evidence Gate 在两轮后仍拒答。因此 Semantic 改善了 retrieval，
却没有形成端到端成功；下一步应检查该失败的 rerank 分数、query coverage 和 Gate
阈值，而不是继续调 chunk。

MultiHopRAG 和双语控制的文档都短于 target，Semantic window 数为 0，质量与 Auto
相同。这也证明“小段落不强切”的路径按设计工作。

结论：样本太小，不能宣称 Semantic 有稳定 +8pp 收益；但可以明确否定“强制
Semantic 总是更好”。当前 Auto 的结构优先策略在成本上明显更合理。

## 8. 真实 DeepSeek 会话验证

使用已有 `Strategy smoke 20260819` 知识库，没有新建测试库：

1. 第一问：`Auto Dense 是什么？`
2. 同一 `conversation_id` 第二问：`它默认在多少个 chunk 后切换到 HNSW？`

指代测试的第二问结果：

- 加载 1 个历史轮次；
- canonical query：`Auto Dense 默认在多少个 chunk 后切换到 HNSW？`；
- 分类 single-hop，confidence 0.88；
- original + canonical 两个视角，并发上限 2；
- 24 个原始候选合并为 13 个，最终全局 Top6；
- EvidenceSet 选 4 个；
- 回答正确给出 100,000 chunk；
- Gate audited IDs = generation selected IDs = citation IDs，四者顺序一致；
- 总延迟 15.71 秒。

本轮还修复了两个 trace 问题：DeepSeek 返回 JSON `null` 时不再写成字符串
`"None"`；Query Understanding 现在在 `query_received` 之后执行并记录 latency、
tokens、retry，LLM 调用失败后规则回退也会如实保留 `llm_called=true`。

第一问的回答曾把 `dense_backend=auto` 与 `chunk_strategy=auto` 两个同名但独立的配置
轴说成相互影响。生成提示已增加“同名选项不代表耦合，必须有原文支持”的约束。

部署修复后又用完整问题做了最终检查：Query Understanding 耗时 3.29 秒，输入/输出
349/314 tokens，0 retry；canonical query 正常，`clarification_question=null`；答案
没有再混淆两个 Auto 配置轴；Gate/生成/citation ID 仍完全一致。该次总延迟 21.06 秒，
其中 reranker 失败等待占 9.15 秒。

### 小样本 Ragas 回归

复用已有 QASPER 与 MultiHopRAG 知识库各 1 题：

| 指标 | 先前 v0.1 小样本 | 当前侧边版 |
|---|---:|---:|
| gold evidence hit | 100% | 100% |
| 全部 gold 文档被引用 | 100% | 100% |
| context precision | 0.75 | 0.75 |
| context recall | 0.50 | 0.75 |
| faithfulness | 0.65 | 0.667* |
| 平均端到端延迟 | 5.74 s | 11.83 s |

`*` 当前 QASPER 的 faithfulness judge 因 max_tokens 输出不完整而报错，所以 0.667 只
来自 MultiHopRAG 一题，不能与旧均值作严格比较。Ragas judge 与答案生成同为
DeepSeek，也不是独立裁判。确定性 gold 文档指标更适合做阻断式回归；Ragas 只作
诊断信号。机器结果为 `reports/results/live-ragas-current-query-understanding.json`。

## 9. 当前最需要关注的不稳定点

### P0/P1：本地 reranker 长输入 HTTP 500

真实查询的 Qwen reranker 在 7.63～9.15 秒后 HTTP 500，系统按设计降级到 lexical
reranker，答案仍正确，但总延迟被明显拉长。Model Runner 日志出现单条输入超过
physical batch 512 tokens 的错误。这不是 ANN、Chunk 或 DeepSeek 故障。

本轮没有直接把 chunk 截到 400 字符来“修复”，因为简单截断可能删除尾部 gold
evidence，造成难以发现的质量倒退。正确的后续实验应比较：

1. 调高 reranker runner 的物理 batch/context 限制；
2. 使用明确支持更长 pair 输入的 rerank 服务；
3. query-aware passage condensation，但必须检查 gold span 保留率；
4. 无论哪种方案，都记录模型成功率、fallback 率、P95 和 Hit@6。

### P1：Gate 已成为部分题目的瓶颈

Semantic 找到了 QASPER 正确证据却仍拒答，说明“没答出”不一定是 chunk/retrieval
问题。Failure attribution 必须区分：gold 未进 Dense、RRF 丢失、rerank 丢失、
EvidenceSet 丢失、Gate false reject、generation unsupported claim。

### P2：Auto 的结构质量判断仍粗

当前只根据 section 名和长度判断。它能覆盖 TXT/PDF 弱结构，但对“有标题、标题下
仍混杂多个主题”的资料不够敏感，需要失败集驱动的可解释特征或父子 chunk 实验。

### 部署层：宿主机 12434 入口曾掉线

Docker Model Runner 在容器内始终可用，但本轮 macOS 宿主机的
`127.0.0.1:12434` 入口中途不可达；真实评测改在一次性容器内通过
`model-runner.docker.internal` 完成。正式 Docker 服务不受影响，但本地 benchmark
脚本应优先提供容器运行方式或启动前 health check。

## 10. 本轮代码与验证

除侧边版已有改动外，本轮补充：

- benchmark/failure-lab 从新的 `retrieval_merge_rerank` 读取全局 Top6，不再把
  recall-only 的 `retrieval_round.reranked_candidates=[]` 误判为 rerank 全丢；
- 多 query view 的 Dense/BM25/RRF 指标按同一 round 合并去重；
- Docker/.env 暴露会话记忆和召回并发默认值；
- Query Understanding trace 补全耗时、token、retry 和真实 LLM 尝试状态；
- 修复 clarification `null` 序列化；
- 生成提示防止混淆同名配置轴；
- Docker 黑盒契约要求出现 `retrieval_merge_rerank`。

验证结果：Ruff 通过；完整 pytest `46 passed`；Compose 配置通过；真实 Qwen 固定
数据集三组评测完成；真实 DeepSeek 指代消解与 EvidenceSet ID 一致性通过；两条
Ragas 回归完成，其中一项 judge 指标明确报错而没有被静默当零。

机器可读结果：

- `reports/results/public-benchmark-f2134c6-same-runtime.json`
- `reports/results/public-benchmark-current-auto.json`
- `reports/results/public-benchmark-current-semantic.json`
- `reports/results/live-ragas-current-query-understanding.json`

## 最终建议

1. 保持 `chunk_strategy=auto` 为默认，不把强制 Semantic 宣传成质量升级；
2. 当前 Query Understanding 保留，但把新增 LLM 成本纳入 trace 和后续回放；
3. 下一项工程优先级应是修复本地 reranker 长输入失败，而不是继续调 chunk 参数；
4. 为 Gate 增加 gold evidence 已命中但拒答的专项集，单独测 false reject；
5. 后续所有 Chunk A/B 固定 embedding、Dense backend、query planner、reranker 和 Gate，
   一次只改变一个变量。

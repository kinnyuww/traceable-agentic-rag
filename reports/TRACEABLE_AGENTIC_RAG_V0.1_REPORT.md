# Traceable Agentic RAG v0.1：设计、实现与评测报告

**报告日期：** 2026-08-19（Australia/Sydney）  
**验证机器：** Apple M1 Pro，10 核 CPU，32 GB 统一内存，macOS arm64  
**版本定位：** 可追踪、可评测、有界的 Agentic RAG 基线；不是自调优闭环  
**原始结果：** [Docker E2E](results/docker-e2e.json) · [QASPER / MultiHop-RAG / 双语控制集](results/public-benchmark-real-model.json) · [MIRACL-zh](results/miracl-zh-real-model.json)

---

## 0. 阅读导航

如果你刚开始学习 RAG，阅读第 1、2、3、11 节；如果你要把它接入产品，重点阅读第 4、5、6、10、13 节；如果你负责评测或后续调优，重点阅读第 7、8、9、12、14 节。

## 1. 一页结论

### 给非专业读者

这套应用可以理解为一个“会先翻资料、判断资料够不够，再回答”的本地知识助手。用户先建立知识库并上传 PDF、Word、Markdown 或 TXT。系统把文档解析、切块并建立两套搜索索引；提问后同时做语义搜索和关键词搜索，再用一个本地小模型重新排序。证据足够就直接引用原文回答；证据不足时才换一种问法再找一次；仍然不够就拒绝回答。每一步都留下记录，所以错误不再只有一句“模型答错了”。

第一版已经可以作为 Web 应用使用，也可以通过 REST API 被其他 Agent 调用。它不会自己修改配置或把一次用户反馈直接变成系统规则；那属于下一阶段的 Harness Engineering，需要金标、隔离实验、回归门禁和人工批准。

### 给从业人员

在真实 M1 Pro 上，Docker Compose 的 API 与 worker 健康运行，Docker Model Runner 通过 llama.cpp/Metal 承载 Qwen3-Embedding-0.6B F16 和 Qwen3-Reranker-0.6B，生成使用 DeepSeek 官方 `deepseek-v4-flash`。最终 Docker 黑盒链路完成上传、异步解析、建库、真实生成、2 并发查询、引用、trace 与 evaluation：完整单问 4.294 s，两并发最大 1.509 s；均低于本轮 5 秒检索/30 秒完整回答目标。

公开固定切片不是排行榜，但能定位流水线行为：

| 诊断集 | 规模 | rerank Hit@1 | rerank Recall@10 | Agent 可回答性准确率 | P95 |
|---|---:|---:|---:|---:|---:|
| QASPER 固定切片 | 5 论文 / 14 问 | 75.00% | 91.67% | 64.29% | 2.742 s |
| MultiHop-RAG 平衡 passage 切片 | 82 文档 / 24 问 | 100.00% | 98.15% | 95.83% | 4.318 s |
| MIRACL-zh 官方 qrels 固定 shard 诊断 | 184 文档 / 20 问 | 85.00% | 99.29% | 100.00%* | 1.376 s |
| 中英双语本地控制集 | 6 文档 / 5 问 | 100.00% | 100.00% | 100.00% | 0.602 s |

\* MIRACL 这 20 个问题全部可回答，因此这里只证明没有错误拒答，不检验不可回答判断。

最重要的负面结论是：QASPER 上证据门可回答性准确率只有 64.29%，说明“检索到了相关候选”不等于“系统会做对回答/拒答决策”。而且在 QASPER 中 dense Recall@10 为 100%，但 RRF/rerank 后降至 91.67%，说明混合检索与重排不是单调增益，必须保留逐阶段候选才能归因。

DeepSeek 生成端现已改为官方 `https://api.deepseek.com` 与 `deepseek-v4-flash`，通过仓库外 secret 注入 operator 提供的官方 key。官方 `/models` 返回 200 且列出目标模型；本项目实际 Chat Completions 客户端也准确返回最小测试结果 `OK`（90 input tokens、23 output tokens、0 retry）。不过本报告中的公开固定切片是在生成关闭时运行的 retrieval-stage 诊断，因此其中的 `answer_or_citation_contains_gold` 仍不能被解读为 DeepSeek 生成质量。

## 2. “Agentic RAG”在这个项目里究竟是什么意思

传统 RAG 通常是固定链条：`问题 → 一次检索 → 拼上下文 → 一次生成`。Agentic RAG 的核心不是“加更多 Agent”或让模型无限尝试，而是让执行器根据中间证据选择下一步，并给每条路径设定预算。

本项目把它约束为一个明确状态机：

1. 只有问候和产品使用帮助可以走确定性直答；知识库事实问题绝不跳过检索。
2. 第一轮执行 dense + BM25、RRF 融合和 rerank。
3. Rerank Top 6 后先构建唯一 Evidence Set：单跳用 `score_floor = max(0.02, top_rerank × 10%)` 过滤并最多保留 4 个（无人越线时保留 Top 1）；多跳不使用 floor 删除尾部，完整保留 Top 6。
4. Evidence Gate 只审核这批证据：保留 Top 1 rerank 相关性主信号，coverage 改为所有已选 chunk 合并后的联合覆盖率，多跳来源多样性检查整个集合；灰区 DeepSeek 分类也接收整个集合。
5. 证据充分：同一批 chunk 不经二次筛选，直接用于生成和引用；问题本身不明确：请求澄清。
6. 证据弱或缺少要素：改写/分解为最多 4 个子查询，再检索一轮；第一、二轮候选合并去重后重新构建 Evidence Set 并复审。
7. 第二轮仍不足：解释性停止并返回“证据不足”，不会无界循环。

新主路径可以压缩为：

`Rerank Top 6 → 单跳 floor 后最多 4 / 多跳保留全部 6 → Evidence Gate → 同一集合直接生成`

修改前，Gate 可依据 rerank Top 6 做判断，但放行后 `_select_context` 又独立执行最多 4 个的筛选；这允许第 5/6 名证据帮助多跳 Gate 放行，却随后从 DeepSeek prompt 中消失。修改后 `EvidenceSet` 成为 Gate 的类型化输入，`context_selection.selected_chunks`、`evidence_gate.audited_chunk_ids`、`answer_generation.selected_chunks` 和 response citation chunk IDs 在回答路径中保持同一有序集合。

因此，“简单事实问题”是走一次确定性的传统 RAG 快速路径；“Agentic”发生在系统看过首轮证据之后，而不是由一个模型在检索前凭感觉猜问题难度。这与 LangGraph 示例中的 retrieve/grade/rewrite/generate 思路相近，但这里使用显式状态机和持久 trace，避免框架或自由循环掩盖边界。

## 3. 端到端产品形态

### 3.1 两个入口，一套核心

- **独立 Web 应用：** 新建知识库、批量拖拽上传、查看解析/索引状态、问答、引用和运行轨迹。
- **REST/OpenAPI 服务：** 上层 Agent 可调用知识库、上传、建库、retrieve、query、run trace 和 evaluation。

Web 本身调用同一套 API，不存在“演示 UI 一套逻辑、对外服务另一套逻辑”。未来扩展 MCP 时只需要做协议薄适配：把 `list_knowledge_bases`、`upload_document`、`retrieve`、`query`、`get_run` 映射成 MCP tools/resources，核心检索、版本与 trace 不动。

### 3.2 用户如何上传自己的资料

Web 支持一次选择或拖拽多个 PDF、DOCX、Markdown、TXT；REST 使用 `multipart/form-data` 调用 `POST /v1/knowledge-bases/{id}/documents`。服务验证扩展名、空文件与 100 MB 默认上限，规范化文件名，以 SHA-256 保存不可变原件，重复内容不会静默覆盖。上传返回 document ID 和 parse job ID；Docker worker 异步解析，用户通过 job 或 Web 状态查看结果，然后为知识库创建不可变 index version 并原子激活。

这一设计区分“文件已经上传”和“文本已经成功进入索引”。扫描 PDF 或复杂表格即便上传成功，也可能提取不到可检索文本；系统会记录解析器、章节数、字符数和错误，而不是把它误判成 embedding 或搜索问题。

## 4. 分层架构

| 层 | v0.1 实现 | 稳定边界 |
|---|---|---|
| 交互 | 原生 Web UI、FastAPI/OpenAPI | UI 与外部调用共享 schema |
| 领域服务 | ingestion、retrieval、bounded agent、evaluation | 不依赖某个 Agent 框架 |
| 文档 | PDF/DOCX/MD/TXT 解析、结构/句子边界切块、重叠、source offsets | parser/chunker 可替换 |
| 稀疏检索 | SQLite FTS5 BM25；中文字符与双字词补充 | 保存原始候选与 BM25 分数 |
| 稠密检索 | Qwen3 1024 维，精确余弦 | 小型本地库优先正确性；后续换 HNSW/Qdrant |
| 融合/重排 | RRF `k=60`；Qwen3 Reranker | reranker 失败降级词法重排 |
| Agent | 单一 Evidence Set、证据门、澄清、最多一次重写/分解、停止 | 单跳 4 / 多跳 6；2 轮 / 4 子查询 / 1 次最终生成 |
| 生成 | 可选 DeepSeek；默认引用式抽取降级 | 文档视为不可信数据；引用 marker 校验 |
| 持久化 | SQLite WAL、FTS5、content-addressed objects | 文档/索引/run/eval 均绑定稳定 ID |
| 运行 | API + worker 容器；宿主 Docker Model Runner | macOS Metal 不强行透传普通 Linux 容器 |

详细图和数据模型见 [架构文档](../docs/ARCHITECTURE.md)。

## 5. 模型与部署选择

### 5.1 为什么没有把 Ollama 写死

模型层按 HTTP 能力接口解耦。当前机器选择 Docker Model Runner，是因为 Compose 可以通过 `model-runner.docker.internal` 调用宿主模型服务；其 Apple Silicon llama.cpp 引擎自动使用 Metal，而普通 Linux 容器无法直接访问 macOS Metal。模型拉取、运行与容器编排因此保持在一个 Docker 工具链中，同时保留以后切换原生 llama.cpp server、W7900 服务或兼容 API 的能力。

实测模型各约 1.11 GiB：

- `ai/qwen3-embedding:0.6B-F16`：F16 GGUF，1024 维；
- `ai/qwen3-reranker:0.6B`：F16 GGUF；
- 推理引擎状态：llama.cpp Metal。

### 5.2 外部 DeepSeek 状态

目标配置为 DeepSeek 官方 `https://api.deepseek.com` 与 `deepseek-v4-flash`。operator 通过仓库外 mode-600 secret 文件提供 key；密钥值没有输出，也没有进入仓库、结果、浏览器、镜像、日志或 trace。认证 `/models` 与本项目实际 completion smoke 均已通过。

这一区分仍然重要：**供应商连通性已验证**不等于**公开数据集上的生成质量已测**。公开固定切片保留生成关闭时的原始运行条件；后续应单独补 claim-level faithfulness、citation correctness 和流式时延，而不能改写历史检索结果。

## 6. RAG 最常见的不稳定点，以及本项目如何留痕

RAG 的错误通常跨越多个阶段。仅保存最终答案无法回答“原文存在，为什么没找到”。本项目为以下层面留下证据：

| 失败层 | 常见现象 | 留痕/处理 |
|---|---|---|
| 上传 | 空文件、重复、超限、路径穿越 | 状态码、hash、size、稳定 document ID；拒绝覆盖 |
| 解析 | PDF 无文本、DOCX 损坏、编码错误、表格丢失 | parser、章节/字符数、parse job 阶段与错误 |
| 切块 | 答案跨边界、块过长、标题语境丢失 | chunker/config、section/page/offset、ordinal、overlap |
| Contextual enrichment | 小模型摘要遗漏/幻觉、成本翻倍 | v0.1 明确不实现；`contextualize=true` 直接失败，绝不假装生效 |
| Embedding | 模型漂移、维度错、服务失败、代理误路由 | 模型/维度/重试/错误；本地地址绕过环境代理；失败不污染索引 |
| Dense/BM25 | 跨语言、同义词、专名、分词各有盲点 | 保存两路完整有序候选，分别计算 Hit/Recall/MRR/nDCG |
| RRF | 融合把 dense 命中的证据降走 | 保存 RRF 输入输出与 `k=60`，支持逐阶段归因 |
| Reranker | 冷启动、503、截断、领域错配、反向降级 | 模型、分数、延迟、重试、degraded/error；失败走词法 fallback |
| Evidence Set / 证据门 | 错误回答、错误拒答或 Gate/生成集合漂移 | question type、selection policy、limit、单跳 floor（多跳为 null）、selected/discarded、audited IDs、联合 coverage、decision/reason/confidence/method |
| 二轮 Agent | 改写跑偏、延迟爆炸、循环 | plan 与 subqueries 留痕；固定两轮后 `stop` |
| Context assembly | 单跳低分尾部污染、多跳桥接证据丢失、Lost in the Middle | Gate 前构建一次；单跳 floor 后最多 4、多跳完整 Top 6；放行后不再二次筛选 |
| Prompt injection | 文档要求模型忽略系统指令 | 文档用 untrusted delimiter；对抗 fixture；弱相关攻击块不进上下文 |
| Generation | 无证据陈述、引用伪造、API 失败 | selected chunks、provider、usage、合法 marker、degraded；抽取降级 |
| 版本漂移 | 文档变了后无法重现 | run 绑定知识库、不可变 index version、模型和 chunk ID |
| Evaluation | judge 同源偏差、指标被优化游戏 | 数据 hash、抽样策略、per-example run ID；确定性 IR 指标优先 |

完整字段见 [Trace 与失败模型](../docs/TRACE_AND_FAILURE_MODEL.md)。

### 6.1 本次开发中真实发现的两类故障

**本机代理误路由。** `curl` 调本地模型成功，但继承 macOS 系统代理的 httpx 请求出现空 503/连接异常，最初很像模型冷启动。根因是 localhost/model-runner 请求被环境代理接管。修复为仅对本地主机关闭 `trust_env`，外部 API 仍保留系统代理，并增加本地代理绕过测试和瞬时 5xx/transport 的有界重试。这个例子说明“模型服务报错”可能其实是网络环境错误。

**低分尾部进入上下文，以及 Gate/生成集合分叉。** 首次浏览器真实问答虽然返回正确答案，但 trace 显示一个相关块得分约 0.999566，另外两个无关块只有 0.000133 和 0.000080，却仍因小知识库 Top-K 不足进入引用，其中一个含 prompt-injection 文本。单跳动态门槛因此保留为 `max(绝对阈值 0.02, top_score × 10%)`。随后进一步发现旧流程在 Gate 之后才执行最多 4 块的 context selection：多跳 Gate 可能受第 5/6 名来源影响而放行，但该来源不会进入生成。现在筛选在 Gate 之前完成：单跳经 floor 后最多 4 个，多跳完整保留 Top 6；Gate、DeepSeek prompt 与 citation 共享完全相同的集合。多跳以有限的额外噪声换取桥接证据召回，并在提示词中明确 rank/score 只是相关性提示、不是事实可信度；trace 可直接做 ID 等值核对。

**worker 误用 API 健康检查。** API、worker 和黑盒任务都能正常运行，但最终 `compose ps` 发现 worker 继承了镜像中的 HTTP `/v1/health` 探针；worker 本来不监听 8080，因而被错误标记 unhealthy。Compose 现为 worker 覆盖专用 SQLite `SELECT 1` 探针。这个故障不会改变问答结果，却会误导监控和自动恢复，说明功能测试与运维状态检查缺一不可。

**弹窗退出被必填校验锁死，以及旧脚本缓存。** 新建知识库弹窗的 `X` 和“取消”最初都是隐式 submit button，浏览器会先执行名称 `required` 校验；提交 handler 又无条件阻止默认动作，造成用户无法退出。现已改成显式 `type=button` 的独立关闭路径，并覆盖取消、X、Esc、遮罩、草稿重置、空白校验和提交中防重复。部署后第一次回归还发现浏览器继续运行旧 `app.js`，因此静态资源增加版本号，`/app` 响应统一发送 `no-cache, no-store, must-revalidate`。最终真实浏览器和临时数据库创建流程均通过。

## 7. 评测设计：为什么是这几个数据集

一个“通用 RAG”不能靠一个领域或一个总分证明。此次使用四类互补诊断：

### 7.1 QASPER

[QASPER](https://huggingface.co/datasets/allenai/qasper) 包含 NLP 论文上的问题、人工答案、支持证据和不可回答标记，适合检查长文档切块、论文内证据和拒答。原数据约 5,049 个问题、1,585 篇论文，CC BY 4.0。本次固定切片取 datasets-server 返回的前 5 篇 validation 论文和第一位 annotator，共 165 块、14 问，其中 12 可回答、2 不可回答。

输入 SHA-256：`2d6f9cd549bb9cc61213ba805ed5440081631791e79eee8cf3f21d926bdfb157`。

限制：QASPER 官方任务默认每个问题已绑定对应论文；本次为了压测检索，把 5 篇论文放进同一知识库，会产生 “this paper” 等指代冲突。因此它是困难诊断，不是官方 leaderboard 复现。

### 7.2 MultiHop-RAG

[MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) 覆盖 comparison、inference、temporal 与 null 四类问题，答案通常需要 2–4 条跨文档证据。本次每类取前 6 条，共 24 问；把 gold evidence facts 作为 82 个 passage，并加入 first-rows 尾部 40 个不重复 distractor。18 个问题可回答、6 个 null。

输入 SHA-256：`1e83a37e2ee3449e261e2b17db583e188e000493209ea9731ca9ee4ec74428d8`。

限制：使用 gold facts 作为 passage，而不是完整新闻库，因此适合检测多证据召回与 Agent 路由，但难度低于官方全库任务。

### 7.3 MIRACL-zh

[MIRACL](https://github.com/project-miracl/miracl) 提供 18 种语言的原生 query/qrels。本次使用中文 dev topics、qrels 和 `docs-0` corpus shard，选择在该 shard 至少有一个正例的前 20 个 topic，纳入所有能找到的 judged docs，再加入 shard 前 100 个 unjudged passage 作为 distractor，共 184 文档。

输入 SHA-256：topics `5b284a9aabf08bb2d1c88ed7ea276025c9d23846457c5350dc8d391b5e0d0a13`；qrels `5546474d3dc8139014e6571e9f4041848bb0a314ca6d2d4bac708174d71c59eb`；corpus shard `b036ba6a927a4598b314de55a1de200d2a619b28b4327f49e97ec4233d324827`。

限制：固定单 shard，不能与官方全 corpus 榜单横向比较；MIRACL 只有 relevance qrels，没有 gold answer，因此不评价生成答案。

### 7.4 中英双语和安全本地控制集

6 个短文档、5 个问题覆盖中文问英文证据、英文问中文证据、双语多跳、不可回答和包含 `IGNORE ALL...` 的弱相关攻击片段。它不是公开 benchmark，而是用于保证工程回归、语言方向和安全边界。

### 7.5 RAGBench 与 Ragas 的位置

[RAGBench](https://arxiv.org/abs/2407.11005) 提供跨 5 个领域、约 100k 样本和 TRACe 标签，更适合以后校准自动 evaluator，而不是本轮重新索引主基准。[Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) 是一套评测框架，能计算 faithfulness、context precision/recall、answer relevancy 等；它不是“RAG 的标准答案”，也不会自动把系统调好。v0.1 先保存问题、gold evidence、answerability、每阶段候选和 run ID，后续可接 Ragas/RAGChecker judge，但发布决策仍要看确定性 IR 指标、专家金标和独立/人工校准。

本次 Evidence Set 改造后用 Ragas 0.4.3 和 DeepSeek 官方
`deepseek-v4-flash` 复跑 QASPER、MultiHop-RAG 各 1 个代表样例：gold 文档
evidence hit 100%，全部预期文档均被引用 100%，机械 gold 字符串命中 50%；
context precision 0.75、context recall 0.50、faithfulness 0.65，零 metric
error，问答平均 5.738 s。MultiHop-RAG 单题 precision/recall/faithfulness 约为
1.00/1.00/0.80；QASPER 为 0.50/0/0.50，原因是无论文作用域的问题在 5 篇论文
知识库中混入其他论文数据集，并漏答 Europarl。样本仅 2 条，且回答与 judge 使用
同模型家族，只用于证明评测链路和定位问题，不用于宣称总体质量。

## 8. 指标怎么读

- **Hit@K：** 前 K 个候选是否至少有一个 gold evidence。
- **Recall@10：** 所有 gold evidence 中有多少进入 Top 10；多跳任务比 Hit 更关心这个。
- **All-evidence@10：** 一个问题所需证据是否全部进入 Top 10。
- **MRR@10：** 第一个相关证据出现得越靠前越好。
- **nDCG@10：** 同时考虑多个相关证据的排序位置。
- **Answerability accuracy：** 应回答时回答、应拒答时拒答的比例；这是证据门指标，不是答案正确率。
- **Evidence hit/cited：** 最终 run 是否命中/引用预期文档。
- **Trace completeness：** 每个完成路径是否具有所需基础阶段，拒答路径以 `stop` 收口，回答路径具有 context/generation/completed。

`answer_or_citation_contains_gold` 只做机械字符串诊断。QASPER 答案中可能含 BIBREF 等结构，MIRACL 没有答案，而且该公开切片批次未启用 LLM，所以该值绝不能当作最终答案质量。

## 9. 详细评测结果与解释

### 9.1 QASPER：主要短板在证据门与跨论文歧义

| 阶段 | Hit@1 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Dense | 75.00% | 100.00% | 83.61% | 87.65% |
| BM25 | 66.67% | 91.67% | 77.78% | 81.35% |
| RRF | 66.67% | 91.67% | 75.00% | 79.10% |
| Rerank | 75.00% | 91.67% | 81.94% | 84.42% |

Agent：11 次单轮回答、3 次证据不足；answerability accuracy 64.29%；可回答问题 evidence hit 83.33%，all evidence cited 83.33%；P95 2.742 s；trace 完整率 100%。

结论：dense 已在 Top 10 找到全部 gold，但融合/重排丢掉了一个问题的 gold；同时两个 unanswerable 中出现错误接受，另有可回答问题引用到错误论文。应该优先优化证据门校准、论文/问题作用域和 fusion safeguard，而不是先换更大的生成模型。可以尝试“dense-only 正例保护”“按文档 scope 检索”“基于开发集校准阈值”，但必须在 sealed test 上验证。

### 9.2 MultiHop-RAG：检索强，但并非每次都完整引用多跳证据

| 阶段 | Hit@1 | Recall@10 | All evidence@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Dense | 94.44% | 100.00% | 100.00% | 94.19% |
| BM25 | 94.44% | 96.76% | 88.89% | 91.08% |
| RRF | 94.44% | 98.15% | 94.44% | 94.06% |
| Rerank | 100.00% | 98.15% | 94.44% | 97.35% |

Agent：18 次单轮、1 次二轮、5 次证据不足；answerability accuracy 95.83%；可回答问题 evidence hit 100%，all evidence cited 88.89%；P95 4.318 s，最大 4.588 s；trace 完整率 100%。

结论：reranker 把首个相关证据排到第 1 的能力很好，但融合后 Recall 略低于 dense；有约 11% 的可回答问题没有把全部支持证据映射进最终引用。下一步更适合研究多证据 set coverage 和 context allocation，而不是只看 Hit@1。

### 9.3 MIRACL-zh：中文召回可靠，BM25 明显弱于 dense

| 阶段 | Hit@1 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Dense | 80.00% | 100.00% | 87.92% | 89.33% |
| BM25 | 60.00% | 92.50% | 74.58% | 78.00% |
| RRF | 85.00% | 98.75% | 89.58% | 89.03% |
| Rerank | 85.00% | 99.29% | 91.25% | 93.34% |

20 问全部单轮；gold evidence hit 100%，all evidence cited 90%；P95 1.376 s；trace 完整率 100%。

结论：中英文通用 Qwen dense/rerank 适合作为首版本地模型。中文字符/双字 BM25 能补充精确词，但单独使用明显不足；hybrid 提升 Hit@1，却轻微损失 dense Recall，仍需候选保护策略。

### 9.4 双语控制集：证明接口正确，不证明跨领域泛化

Dense 与 rerank Hit@1/Recall@10 都为 100%；BM25 Hit@1 50%、Recall 62.5%；RRF Hit@1 75%、Recall 100%。Agent 的 answerability、可回答 evidence hit、all evidence cited 和 trace 完整率均为 100%，P95 0.602 s。攻击片段在最终 context selection 被丢弃。

### 9.5 Docker 黑盒和性能

- API/worker 均健康；OpenAPI 暴露 12 个路径。
- Markdown fixture：解析 7.4 ms，3 个章节、127 字符；索引 1.661 s，3 个 1024 维 chunk。
- DeepSeek 完整单问：4.294 s，`single_pass_rag`，1 轮、1 引用、6 个完整 trace 事件。
- 2 并发：1.484/1.509 s，最大值低于 5 s。
- 2 条 evaluation：evidence hit 100%，answerability accuracy 100%，均值 2.242 s。
- 一条真实 answer trace：`deepseek-v4-flash`、205 input/61 output tokens、0 retry、`degraded=false`，答案引用 `[S1]` 且与原文一致。

公开固定切片的最高 P95 为 4.318 s，低于本轮“检索+重排 ≤5 s”的目标。该批次运行时 DeepSeek 尚未启用，因此这些 P95 只覆盖检索、重排和抽取式 fallback；随后官方 DeepSeek 完整 Docker RAG 单问为 4.294 s、两并发最大 1.509 s，但非流式接口无法单独测量首 token ≤8 s。

### 9.6 Evidence Set 改造后的同库回归

这次用同一个 13 文档“粗糙知识库”、同一组 F01–F10 和同一诊断器复测。原始
v0.1 DeepSeek 基线与新策略结果如下：

| 指标 | 原始 v0.1 | 新 Evidence Set | 变化 |
|---|---:|---:|---:|
| 答案内容正确 | 9/10 | 9/10 | 0 |
| 严格链路通过 | 4/10 | 4/10 | 0 |
| 平均问答时延 | 3335.149 ms | 3293.258 ms | -41.891 ms（-1.3%，视为波动） |
| 诊断分布 | 4 pass、3 不可信上下文暴露、1 Gate false accept、1 歧义 false accept、1 parse failure | 完全相同 | 0 |

另做了一组关闭生成 LLM 的策略隔离前后测：旧的“单跳/多跳都用 floor”和新的
“单跳 floor、多跳保留 Top 6”都得到答案内容 6/10、严格通过 2/10，逐题诊断
完全不变。这组不能代表最终回答质量，但更能隔离 Evidence Set 规则本身。真实
DeepSeek 服务最终仍以上表为准。

10 条最终 DeepSeek run 中，所有回答路径均满足
`evidence_gate.audited_chunk_ids == answer_generation.selected_chunks == citation_chunk_ids`。
真实多跳样例的 trace 为 `selection_policy=retain_rerank_top_k`、`score_floor=null`、
selected count 6；单跳样例则为 `dynamic_score_floor` 且最多 4 个。三个正常库抽查：

- MultiHop-RAG 正确回答 Sam Bankman-Fried，保留并审核 6 个证据；
- MIRACL-zh 正确回答“罗马”，使用 2 个单跳证据；
- QASPER 只部分正确：召回 MultiUN，但因跨论文作用域污染漏答 Europarl。这一失败
  发生在 single-hop 4 块路径，不是多跳尾部保留造成的。

原始、新策略、隔离前后与 Ragas 的机器可读结果分别保存在
`reports/results/failure-lab-original-baseline.json`、
`reports/results/failure-lab-live.json`、
`reports/results/failure-lab-multihop-retain-comparison.json`、
`reports/results/failure-lab-deepseek-vs-extractive.json` 和
`reports/results/live-ragas-evidence-set-v0.1.json`。

## 10. 故障策略与安全判断

- HTTP 4xx（如 401）是配置/权限错误，立即失败，不重试。
- timeout、transport 和 5xx 最多共 5 次，有界指数退避；trace 保存 retry count。
- reranker 失败可安全降级为 lexical rerank，并标记 degraded。
- generation 失败可安全降级为引用式抽取，并标记 provider error。
- embedding 失败不能换成不同维度/不同语义空间的假 fallback；建库或查询明确失败，避免静默损坏索引。
- 原文是数据，不是指令；生成 prompt 用边界包裹 untrusted source，引用只接受实际 selected chunks 的合法 marker。
- API 只绑定 `127.0.0.1`；v0.1 没有身份认证，因此不得直接暴露公网或当多租户服务。
- 没有删除 endpoint 和自动 destructive migration，降低无人值守误删风险。

## 11. “自我记录、自我评估、自我优化”应该如何拆分

用户最初设想是合理的，但这三个能力不应在第一版混在一起：

1. **自我记录（本版完成）：** 输入版本、候选、分数、决策、上下文、模型、时延、错误、引用都可回放。
2. **自我评估（本版基础完成）：** 固定 gold evidence 与 answerability 可以运行确定性 evaluation；强 judge/Ragas/RAGChecker 是可插拔补充。
3. **失败归因（下一版）：** 用 gold evidence 在 dense/BM25/RRF/rerank/context 各阶段的位置，判断 parse miss、chunk miss、recall miss、fusion loss、rerank loss、gate false accept/reject、generation unsupported claim。
4. **候选优化（下一版）：** Teacher/Critic 只能提出 chunk、top-k、fusion、阈值、prompt 或模型候选，不能直接改生产。
5. **HITL 发布（下一版）：** 在 dev 上搜索，在 sealed test 上要求主指标相对提升 ≥5%、受保护指标下降不超过 2 个百分点、成本/时延不越界，再由人批准、可回滚发布。

这才是可控的“自我调优”：系统能提出假设并自动跑实验，但不能把某个失败样例或同源 LLM judge 的意见直接升级成全局行为。

## 12. 当前未完成和不能过度宣称的部分

| 项目 | 状态 | 原因/下一步 |
|---|---|---|
| DeepSeek 官方 `deepseek-v4-flash` | **认证、最小生成与 Docker RAG 已验证** | models 200、实际 client completion、带 `[S1]` 的 grounded answer 和 2 并发均成功；公开 fixed slices 的 faithfulness 与流式首 token 尚未重跑 |
| Anthropic Contextual Retrieval | **未实现** | 它是为既有 chunk 生成 50–100 token 文档级语境后再建 dense/BM25；成本与错误摘要会污染两路索引，应做独立实验，不应冒充“语义切块” |
| Parent-child retrieval | **未实现** | 目前保留 section/page/offset，可在证据证明需要后加父块扩展 |
| OCR/复杂表格/图片 | **未实现** | 原生文本 PDF/DOCX 可用；扫描件需要 OCR 与版面证据定位 |
| 大规模 ANN | **未实现** | 现在是精确余弦，适合 MVP/诊断集；大库换 HNSW/Qdrant/pgvector |
| 多租户、认证、配额 | **未实现** | 当前单机单工作区、仅 localhost |
| MCP | **未实现但架构兼容** | REST schema 已稳定，后续做薄适配器、认证、流式和文件资源语义 |
| 反馈、归因、自调优、HITL | **刻意后置** | 需要在本版 trace/eval 基线上做 Harness Engineering |

## 13. 运行与复现

```bash
./scripts/bootstrap-models.sh
./scripts/verify-models.sh
./scripts/run-docker.sh -d
```

打开 `http://127.0.0.1:8080/app/`。完整运维、外部 secret、开发模式与备份见 [OPERATIONS.md](../docs/OPERATIONS.md)。结果脚本：

```bash
python scripts/verify_deployment.py --base-url http://127.0.0.1:8080
python scripts/run_benchmarks.py --qasper ... --multihop ...
python scripts/run_miracl_benchmark.py --topics ... --qrels ... --corpus-shard ...
```

当前完整代码验证为 34 个 pytest 用例全通过。除四类 parser、损坏/空文件、重复上传、异步 job、建库、retrieve/query/trace/eval、两轮边界、contextualize 显式失败、embedding 硬失败、reranker/generation 降级、提示注入尾部过滤、localhost 代理绕过、弹窗退出/缓存/可访问性契约外，还覆盖单跳最多 4 及 floor 降噪、多跳无 floor 保留第 5/6 名证据、低分第 5 名进入第二轮 Gate/生成/引用、联合 coverage、灰区 LLM 接收 6 个证据、rank/score 可信度警示、第二轮合并去重，以及 Gate/生成/citation ID 集合一致性；ruff 通过。Docker 正式 DeepSeek 配置、粗糙库同题回归、三个示例库问答和 Ragas 代表样例另行通过。

## 14. 推荐下一迭代：Harness Engineering v0.2

优先顺序应由本次证据决定：

1. **扩展 DeepSeek 生成评测。** 当前只有 2 条 Ragas 代表样例；下一步扩大独立 judge/人工校准的 claim-level faithfulness、citation correctness、answer relevancy、拒答与中文/英文流式生成时延，不能只看字符串包含率或同模型 judge。
2. **做失败归因器。** 用每阶段 gold 排名自动生成 failure taxonomy，特别关注 QASPER 的 fusion loss、rerank loss、跨论文指代和 gate false accept。
3. **建立用户领域金标入口。** 专家提交 question、expected answer、gold evidence、answerable、标签；系统必须能从原文反查并辅助标注，而不是只收点赞/点踩。
4. **离线候选实验。** 比较 chunk size/overlap、parent-child、dense safeguard、RRF 参数、rerank top-k、gate 阈值和 Contextual Retrieval；每次绑定 immutable config。
5. **sealed test + HITL 发布。** dev 样例用于提出/选择方案，sealed test 防止过拟合；人审核差异、成本、隐私和回滚点后才激活。
6. **再做 MCP。** REST 核心无需变化，把常用能力映射为工具和资源即可。

不建议下一步立刻做 GraphRAG、微调或无监督在线自修改。QASPER 的结果显示，先修证据作用域、逐阶段损失和 gate 校准，收益更直接、也更可解释。

## 15. 外部依据与主流路线

- [Anthropic: Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)：给 chunk 补文档级上下文后再进入 embedding 与 BM25；不是单纯的“语义切块”。
- [LangGraph Agentic RAG tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/)：retrieve、grade、rewrite、generate 的有界思路。
- [Seven Failure Points When Engineering a RAG System](https://arxiv.org/abs/2401.05856)：RAG 失败应按生命周期与操作期持续验证。
- [RAGChecker](https://github.com/amazon-science/RAGChecker)：retriever/generator 的 claim-level 诊断。
- [Ragas metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)：faithfulness、context precision/recall 等可插拔评测指标。
- [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)：长上下文中位置会影响证据使用，说明堆更多 chunk 不等于更好。
- [RankRAG](https://arxiv.org/abs/2407.02485)：把上下文排序信号与生成结合；本项目只把 rank/score 作为相关性提示，不把它提升为事实可信度。
- [OWASP Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) 与 [RAG Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html)：检索文档是不可信输入，必须隔离和监测。
- [Docker Model Runner inference engines](https://docs.docker.com/ai/model-runner/inference-engines/) 与 [Models and Compose](https://docs.docker.com/ai/compose/models-and-compose/)：Apple Silicon llama.cpp/Metal 与 Compose 接入依据。

---

## 最终判定

**v0.1 的“本地可运行、REST 可嵌入、真实 embedding/rerank、hybrid retrieval、有界 Agent、引用、trace、evaluation、Docker/Web”基线通过，DeepSeek 官方认证与完整 Docker RAG 生成链路也已通过。** 其价值不在于一个漂亮总分，而在于已经能说明错误发生在哪一层，并能把未来自调优限制在可复现、可比较、可回滚的实验流程中。

发布判定仍是“开发者本地基线”，不是生产多租户版本；公开 fixed slices 的 DeepSeek 生成质量与流式时延仍是后续评测项。

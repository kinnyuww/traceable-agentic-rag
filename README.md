# Traceable Agentic RAG

> v0.1 Baseline — local-first, traceable, evaluable, and usable as both a Web
> RAG application and an embeddable REST service.

一个本地优先、可独立使用、也可通过 REST 嵌入其他 Agent 的可追踪
Agentic RAG。v0.1 的重点不是声称“自动进化”，而是先把数据、索引、检索、
路由、证据、回答与评测变成可复现的工程基线。

已在 Apple M1 Pro 32 GB 上完成真实模型和 Docker 验证：文档上传与解析、
结构化切块、不可变索引版本、dense + BM25 + RRF 混合检索、Qwen rerank、
有界二轮 Agentic 检索、引用回答、逐阶段 trace、离线评测和 Web UI 均可运行。

## v0.1 的定位

知识库事实问题永远先检索，不允许生成模型因为“似乎知道答案”而跳过 RAG。
首轮证据充分时走低延迟单轮路径；证据弱、缺少问题要素或需要多跳时，才改写/
分解问题并再检索一次。系统最多检索 2 轮，第二轮最多 4 个子查询，最后回答、
请求澄清或明确拒答，没有无界 Agent 循环，也不会在线修改自己的配置。

这一版比较有辨识度的不是又包装了一条 RAG pipeline，而是：

- **结果与过程分开评测**：粗糙知识库中答案文本正确率为 90%，但完整链路干净率
  只有 40%，能在“暂时答对”时提前发现错误引用、提示注入暴露和路由失真。
- **检索每一层都可见**：dense、BM25、RRF、rerank、证据门、上下文筛选、生成与
  引用都保存候选文件、章节、分数、时延、模型、重试和不可变版本 ID。
- **应用与服务是同一个产品**：人可以用 Web UI，Agent 可以调用 REST；两者共享
  同一数据库、对象存储、索引、RAG Core 和 run trace。
- **Agentic 行为有边界**：什么时候第二轮、为什么停止、为何澄清或拒答都有明确
  状态和预算，适合后续做失败归因与受控实验。

## 五分钟启动

当前验证目标为 Apple Silicon Mac、Docker Desktop，以及启用了 host-side TCP
的 Docker Model Runner。

```bash
./scripts/bootstrap-models.sh
./scripts/verify-models.sh
./scripts/run-docker.sh -d
```

启动后：

- Web 应用：<http://127.0.0.1:8080/app/>
- REST/OpenAPI：<http://127.0.0.1:8080/docs>
- 健康检查：<http://127.0.0.1:8080/v1/health>

数据保存在 Docker volume `traceable-rag-agent-data`。真实 API key 只通过仓库外
secret 文件或进程环境注入，不能写入仓库、镜像、浏览器或 trace。

## 三种使用路线

### 1. 全 Web UI

在网页中新建知识库，上传 PDF、DOCX、Markdown 或 UTF-8 TXT，等待解析完成，
建立并激活索引，然后直接问答、查看引用和运行轨迹。适合个人或领域专家操作。

### 2. Web UI 建库，Agent 通过 API 调用

先由人通过 Web 管理知识库，再让现有 Agent 使用知识库 ID 调用：

```bash
curl -s http://127.0.0.1:8080/v1/query \
  -H 'Content-Type: application/json' \
  -d '{
    "knowledge_base_id": "kb_REPLACE_ME",
    "question": "这份制度中员工每年有多少天年假？"
  }'
```

也可以按 Web UI 中的名称调用可运行示例：

```bash
PYTHONPATH=src uv run --no-sync python examples/rest_workflow.py query \
  --knowledge-base-name '粗糙知识库' \
  --question 'ZX-417 的最高工作温度是多少？'
```

### 3. 全 API 建库与调用

上层程序可以通过 REST 创建知识库、批量上传、轮询解析 job、建立索引、查询并
读取完整 run trace。仓库中的示例覆盖整个生命周期：

```bash
PYTHONPATH=src uv run --no-sync python examples/rest_workflow.py build \
  --name '产品手册' \
  --description '由上层 Agent 通过 REST 创建' \
  --question '安装前需要满足什么条件？' \
  ./manual.md ./faq.pdf
```

三条路线的完整契约和调用顺序见 [使用模式指南](docs/USAGE_MODES.md)。

## 当前切块基线

v0.1 采用确定性的结构感知切块：先按 PDF 页、DOCX/Markdown 标题或文本 section
解析，再在 section 内优先寻找段落/句子边界，目标约 1100 字符、重叠约 160
字符。每个 chunk 保存文档、页码、章节、字符 offset 和 ordinal，并把结构信息
作为本地 `contextual_text` 加入 embedding、BM25 和 rerank。

它不是 LLM 语义切块，也没有把 Anthropic-style Contextual Retrieval 冒充为已
实现功能。现有边界、中文长文本风险和后续候选实验详见
[切块策略说明](docs/CHUNKING_STRATEGY.md)。

## 模型与运行边界

- Embedding：`ai/qwen3-embedding:0.6B-F16`，1024 维
- Reranker：`ai/qwen3-reranker:0.6B`
- 本地推理：Docker Model Runner 的 llama.cpp 引擎；Apple Silicon 使用 Metal
- 生成：OpenAI-compatible 适配器；当前验证配置为 DeepSeek 官方
  `deepseek-v4-flash`
- Dense store：v0.1 使用精确 cosine search，适合本地基线与诊断；大规模部署的
  替换边界是 HNSW/Qdrant/pgvector adapter

生成模型可关闭；失败时系统保留可引用的抽取式降级。配置见
[本地运维指南](docs/OPERATIONS.md)。

## REST 能力

`/v1` 契约包含：

- 创建、列出和读取知识库；
- 批量上传、解析和列出文档；
- 创建不可变索引版本、读取 job 与阶段 trace；
- 单独调用 hybrid retrieval；
- 调用完整 Agentic query，获得引用、route、轮次、run ID 和 trace URL；
- 读取不可变 run trace；
- 提交固定金标评测并读取结果。

未来 MCP 只需把 `query`、`retrieve`、`upload_document`、`get_run` 等工具映射到
这些 schema，不需要重写检索核心。

## 验证与报告

- [多颗粒度框架与心智模型报告](reports/AGENTIC_RAG_FRAMEWORK_MENTAL_MODEL.md)
- [交互式 Agentic RAG 心智模型](reports/explainer-agentic-rag-mental-model.html)
- [真实示例导入与 Ragas 评测报告](reports/LIVE_DEMO_AND_RAGAS_EVALUATION.md)
- [粗糙知识库：故障实验室与逐层定位指南](reports/FAILURE_LAB_GUIDE.md)
- [完整设计与评测报告](reports/TRACEABLE_AGENTIC_RAG_V0.1_REPORT.md)
- [系统架构](docs/ARCHITECTURE.md)
- [不稳定面与 Trace 契约](docs/TRACE_AND_FAILURE_MODEL.md)
- [Docker 黑盒结果](reports/results/docker-e2e.json)
- [QASPER / MultiHop-RAG / 双语控制集结果](reports/results/public-benchmark-real-model.json)
- [MIRACL-zh 结果](reports/results/miracl-zh-real-model.json)

开发验证：

```bash
UV_CACHE_DIR=/tmp/ragagent-uv-cache uv sync --extra dev --no-editable --python 3.12
UV_CACHE_DIR=/tmp/ragagent-uv-cache uv run --no-sync ruff check .
UV_CACHE_DIR=/tmp/ragagent-uv-cache uv run --no-sync pytest -q
```

## Roadmap：从 Baseline 到 Harness Engineering

v0.1 有意不做 OCR/复杂表格、多租户/RBAC、MCP、反馈审批、自动失败归因、
自动参数实验、自调优和 HITL 发布门禁。下一阶段计划围绕下面的闭环展开：

```text
用户/专家反馈
  → 金标答案与证据审批
  → 逐层失败归因
  → 生成单变量候选实验
  → development set 对比
  → sealed regression test
  → 质量 / 成本 / 时延门禁
  → HITL 批准
  → 新索引或配置版本发布 / 可回滚
```

候选实验包括切块、解析、embedding、BM25、fusion、reranker、context selection、
prompt 和 evidence gate。所谓 self-tuning 首先是“离线提出并验证候选”，不是让
线上 Agent 根据一次点赞就直接改生产配置。

后续接入层可以增加 MCP；更后面才考虑多租户、权限、漂移监控和受限自动提升。

## License

Apache-2.0.

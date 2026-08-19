# 当前本地服务：真实示例导入与 Ragas 评测报告

生成日期：2026-08-19  
服务地址：`http://127.0.0.1:8080`  
生成模型：DeepSeek 官方 `deepseek-v4-flash`  
Ragas：`0.4.3`

## 1. 先回答最容易混淆的问题

昨天的公开数据评测**没有实际运行 Ragas**。当时运行的是更可重复的确定性与 trace 指标：Dense/BM25/RRF/Rerank 的 Hit@K、Recall@10、MRR、nDCG，Agent 的 evidence hit、all evidence cited、answerability 和 trace completeness。

本次补上了两件事：

1. 把 QASPER、MultiHop-RAG、MIRACL-zh 的相同固定切片通过当前 Docker 服务的 REST 接口真正导入持久知识库；
2. 用当前服务和 DeepSeek V4 Flash 重新生成答案，并实际运行 Ragas 的 faithfulness、context precision 和 context recall 小样本诊断。

Ragas 不是“RAG 的权威总分”。它适合补充判断生成是否忠于上下文、检索上下文是否相关/完整；gold 文档是否被召回和引用，仍然由确定性指标负责。

## 2. 已导入当前服务的三个知识库

| 网页中的知识库 | 文档 | 金标问题 | Active index | 原生抽样评测 |
|---|---:|---:|---|---|
| 示例 · QASPER 论文问答 | 5 | 14 | `idx_WBSsOA73WVu2` | 3 题，evidence hit 66.7%，answerability 100%，均值 7.305 s |
| 示例 · MultiHop-RAG 多跳推理 | 82 | 24 | `idx_ffcwYWhM-82Z` | 3 题，evidence hit 100%，answerability 100%，均值 4.400 s |
| 示例 · MIRACL 中文检索 | 184 | 20 | `idx_TdobKnl2LeVt` | 3 题，evidence hit 100%，answerability 100%，均值 3.654 s |

这些知识库不是直接写 SQLite 创建的。导入脚本调用了与网页相同的公开接口：

```text
POST knowledge-bases
  → multipart 上传文档
  → worker 解析每个文档
  → POST index-builds
  → Qwen3 Embedding + FTS5
  → 激活不可变 index version
  → POST evaluations
  → DeepSeek 真实 query/run/trace
```

因此它们可以用来理解整个产品，而不只是查看一份离线 JSON。

## 3. 原始数据已保存在哪里

所有公开输入保存在仓库的 gitignored `data/benchmark-sources/`，共约 93 MB：

| 文件 | SHA-256 |
|---|---|
| `qasper-validation-first-rows.json` | `2d6f9cd549bb9cc61213ba805ed5440081631791e79eee8cf3f21d926bdfb157` |
| `multihoprag-train-first-rows.json` | `1e83a37e2ee3449e261e2b17db583e188e000493209ea9731ca9ee4ec74428d8` |
| `miracl-zh-dev-topics.tsv` | `5b284a9aabf08bb2d1c88ed7ea276025c9d23846457c5350dc8d391b5e0d0a13` |
| `miracl-zh-dev-qrels.tsv` | `5546474d3dc8139014e6571e9f4041848bb0a314ca6d2d4bac708174d71c59eb` |
| `miracl-zh-docs-0.jsonl.gz` | `b036ba6a927a4598b314de55a1de200d2a619b28b4327f49e97ec4233d324827` |

哈希与昨天报告中的输入完全一致。每个知识库的 dataset key、当前 document ID、问题、参考答案、gold document、索引和原生评测 ID 保存在：

- `data/demo-manifests/qasper.json`
- `data/demo-manifests/multihoprag.json`
- `data/demo-manifests/miracl_zh.json`

## 4. 现在应该怎样体验

打开：<http://127.0.0.1:8080/app/>

如果页面此前已经打开，刷新一次。新版页面在每个示例知识库的问答欢迎区域增加了“示例问题”：

1. 从左侧选择一个以“示例 ·”开头的知识库；
2. 点击一个示例问题，它只会填入输入框；
3. 检查问题无误后点击“发送”；
4. 先看答案下方的 `route`、轮次、耗时和引用；
5. 再打开“运行轨迹”，依次查看 `retrieval_round`、`evidence_gate`、`context_selection`、`answer_generation`。

### 4.1 第一站：MIRACL 中文检索

推荐问题：

```text
意大利首都是哪里？
```

观察重点：

- 中文问题是否一轮找到正确中文文档；
- Dense、BM25 和 Rerank 的候选次序是否不同；
- Evidence Gate 为什么直接 `answer`；
- 最终引用是否只保留强相关 chunk。

### 4.2 第二站：MultiHop-RAG 多证据

推荐使用页面中的第一个问题，即 Sam Bankman-Fried 的跨 The Verge、TechCrunch 多证据问题。

观察重点：

- 最终答案引用 3 个 gold evidence 文档；
- 本次仍是 `single_pass_rag`，说明“多跳问题”不等于“必定运行两轮”：如果第一轮已经找到足够且来源多样的证据，就不浪费第二轮；
- top 4 里仍有一个分数较高的 distractor，但 context selection 与引用需要把它和 gold evidence 区分开。

### 4.3 第三站：QASPER 的真实失败

推荐问题：

```text
which datasets did they experiment with?
```

参考答案是 `Europarl | MultiUN`。当前运行却混入了另两篇论文的 `DL-PS / EC-MT / EC-UQ`，并遗漏 `Europarl`。

这是一个非常有价值的失败样例：QASPER 原任务的问题作用域是“当前论文”，但导入时 5 篇论文放在同一个知识库，问题里的 `they` 没有携带 paper scope。系统确实引用到了 gold 文档，所以 evidence hit 为真；但同时召回并采用了错误论文，最终答案仍然错。

这证明：

- evidence hit 100% 不等于答案正确；
- “找到一份正确材料”不等于“排除了错误材料”；
- 下一步需要 document scope/metadata filter、问题作用域或 context precision 保护，而不是单纯换更大生成模型；
- Evidence Gate 对这种跨文档歧义存在 false accept，目前 route 仍是 `single_pass_rag`。

## 5. 当前服务的 Ragas 实测

Ragas 使用当前服务的 `/v1/query` 生成答案、`/v1/retrieve` 取得 top-6 context，再由 DeepSeek V4 Flash 作为 judge。代表性样本为 2 条：MultiHop-RAG 1 条、QASPER 1 条。

### 5.1 汇总

| 指标 | 结果 |
|---|---:|
| 样本数 | 2 |
| Gold evidence hit | 100% |
| All expected evidence cited | 100% |
| 答案或引用包含参考答案 | 50% |
| 平均端到端回答延迟 | 5.410 s |
| Ragas Faithfulness | 0.625 |
| Ragas Context Precision with reference | 0.750 |
| Ragas Context Recall | 0.750 |
| Ragas metric errors | 0 |

### 5.2 分样本

| 数据集 | Gold 字符串 | Faithfulness | Context precision | Context recall | 解释 |
|---|---:|---:|---:|---:|---|
| MultiHop-RAG | 命中 | 0.75 | 1.00 | 1.00 | 三个 gold 文档全部引用，回答给出 Sam Bankman-Fried；judge 仍认为部分原子陈述未完全获得支持 |
| QASPER | 未命中 | 0.50 | 0.50 | 0.50 | 召回与答案混入其他论文，遗漏 Europarl；这是检索作用域和上下文精度问题 |

完整逐题回答、引用、run ID、top hits、Ragas 结果和错误字段在：

`reports/results/live-ragas-deepseek-v4-flash.json`

### 5.3 Ragas 自身也需要工程治理

第一次运行 Ragas 时，默认 judge 输出上限 1024 token 导致结构化结果被截断，产生 `IncompleteOutputException`。本次将 `max_tokens` 提高到 4096 后，同一组指标全部成功。

另外，`ragas==0.4.3` 与当时解析到的 `langchain-community==0.4.2` 出现缺失 `chat_models.vertexai` 的导入不兼容；隔离环境固定为 `langchain-community==0.3.31` 后恢复。

这两个问题进一步说明：评测器也可能失败。报告必须区分“0 分”“无分数”和“评测器异常”，不能把异常吞掉。

## 6. 为什么没有给 MIRACL 做 Ragas 答案分

MIRACL 的 gold 是 query-to-document relevance qrels，不包含 gold answer。它非常适合评测 Dense/BM25/RRF/Rerank 的 Hit、Recall、MRR 和 nDCG，但不适合伪造参考答案后再算 answer correctness/context recall。

因此本轮 MIRACL 保持两种诚实口径：

- 当前 Docker 服务：3 条真实 DeepSeek 抽样，evidence hit 与 answerability 均为 100%；
- 昨天固定 20 问 retrieval 诊断：Rerank Hit@1 85%，Recall@10 99.29%，trace completeness 100%。

前者证明当前服务调用链可用，后者才是较完整的中文检索统计；二者不能合并成同一个 Ragas 总分。

## 7. 如何复现

### 7.1 重新导入三个示例知识库

```bash
UV_CACHE_DIR=/tmp/ragagent-uv-cache \
PYTHONPATH=src:scripts \
uv run --no-sync python scripts/load_demo_datasets.py --run-native-eval
```

脚本发现同名、文档数一致且已有 active index 的示例知识库时会复用 manifest，避免重复导入。

### 7.2 重新运行 Ragas 小样本

Ragas 被放在 `/tmp` 隔离环境，不进入应用容器，也不污染主项目依赖：

```bash
UV_CACHE_DIR=/tmp/ragagent-ragas-uv-cache \
uv venv /tmp/ragagent-ragas-venv --python 3.12

UV_CACHE_DIR=/tmp/ragagent-ragas-uv-cache \
uv pip install --python /tmp/ragagent-ragas-venv/bin/python \
  ragas==0.4.3 langchain-community==0.3.31 openai httpx

set -a
source "$HOME/.config/traceable-rag-agent/secrets.env"
set +a
RAGAS_DO_NOT_TRACK=true \
/tmp/ragagent-ragas-venv/bin/python scripts/evaluate_live_ragas.py
```

脚本不会打印 API key。默认只评 QASPER 和 MultiHop-RAG 各 1 条，控制成本；可通过 `--examples-per-dataset` 扩大，但扩大前应先确定费用、限流和独立 judge 策略。

## 8. 如何理解三类评测

| 评测层 | 回答什么 | 本项目中的实现 |
|---|---|---|
| 确定性检索指标 | Gold evidence 有没有被找到、排在第几？ | Hit/Recall/MRR/nDCG，各检索阶段可归因 |
| 原生 Agent evaluation | 最终引用是否碰到 gold 文档，应答/拒答是否正确？ | `/v1/evaluations`，每例保存 run ID |
| Ragas/LLM judge | 答案是否忠于上下文、上下文是否相关/覆盖参考答案？ | 当前外部 adapter，小样本、错误显式记录 |

一个合格发布门不应只看其中一类。特别是 QASPER 已经证明：gold 文档被引用、answerability 路由正确，答案仍可能因混入错误文档而失败。

## 9. Ragas 分数的限制

- 答案和 judge 都使用 DeepSeek V4 Flash，不是独立裁判，可能有同模型偏差；
- 只有 2 个代表性英文样本，不能作为总体质量置信区间；
- Faithfulness 使用 API 返回的 citation excerpt；context 指标使用另一次 `/v1/retrieve` 的 top-6，不完全等同于 Agent 实际第二轮上下文；
- Ragas prompt 本身也是可调超参数，改 prompt 可能改变分数；
- LLM judge 适合诊断，不应单独决定自动发布或自调优。

## 10. 本次最重要的下一步

与其立刻扩大 Ragas 样本，更值得先修 QASPER 暴露出的作用域问题：

1. 在 query 中增加可选 `document_ids` / metadata filter；
2. 网页允许用户选择“整个知识库”或“当前文档”；
3. Evidence Gate 对代词/“this paper”类问题检查文档歧义；
4. context selection 增加 document coherence 或高分错误来源抑制；
5. 用 QASPER dev/sealed test 比较修复前后 context precision、答案正确率和误拒率；
6. 再把 Ragas judge 换成独立模型或加入人工复核，作为 HITL 发布证据之一。

这样用户看到的不只是“有几个示例”，而是一条可以亲手观察成功、失败、trace 和改进方向的完整学习路径。

## 11. 数据与评测框架来源

- QASPER：<https://huggingface.co/datasets/allenai/qasper>
- MultiHop-RAG：<https://huggingface.co/datasets/yixuantt/MultiHopRAG>
- MIRACL：<https://github.com/project-miracl/miracl>
- Ragas metrics：<https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/>

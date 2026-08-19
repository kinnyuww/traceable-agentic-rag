const state = {
  knowledgeBases: [],
  activeKnowledgeBase: null,
  lastRun: null,
  polling: new Set(),
  creatingKnowledgeBase: false,
  conversationIds: {},
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch { body = {}; }
  if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
  return body;
}

function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.className = `toast visible${error ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { element.className = "toast"; }, 4200);
}

function escapeHtml(value = "") {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

async function checkHealth() {
  try {
    const health = await api("/v1/health");
    $("#health-dot").className = "status-dot ok";
    $("#health-label").textContent = `${health.providers.embedding} · ${health.providers.rerank}`;
  } catch (error) {
    $("#health-dot").className = "status-dot error";
    $("#health-label").textContent = "服务不可用";
  }
}

async function loadKnowledgeBases(preferredId = null) {
  state.knowledgeBases = await api("/v1/knowledge-bases");
  renderKnowledgeBases();
  if (preferredId) selectKnowledgeBase(preferredId);
  else if (!state.activeKnowledgeBase && state.knowledgeBases.length) selectKnowledgeBase(state.knowledgeBases[0].id);
  else if (!state.knowledgeBases.length) showEmpty();
}

function renderKnowledgeBases() {
  $("#kb-list").innerHTML = state.knowledgeBases.map((kb) => `
    <button class="kb-item ${state.activeKnowledgeBase?.id === kb.id ? "active" : ""}" data-kb-id="${kb.id}">
      <strong>${escapeHtml(kb.name)}</strong>
      <span>${kb.document_count} 个文档</span>
    </button>
  `).join("") || '<span class="muted mono">暂无知识库</span>';
  $$(".kb-item").forEach((button) => button.addEventListener("click", () => selectKnowledgeBase(button.dataset.kbId)));
}

async function selectKnowledgeBase(id) {
  state.activeKnowledgeBase = state.knowledgeBases.find((item) => item.id === id) || await api(`/v1/knowledge-bases/${id}`);
  $("#empty-state").classList.add("hidden");
  $("#workspace").classList.remove("hidden");
  $("#workspace-title").textContent = state.activeKnowledgeBase.name;
  updateIndexPill();
  renderKnowledgeBases();
  conversationIdFor(id);
  syncConversationControls();
  renderConversationIntro();
  await loadDocuments();
}

function createConversationId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `conversation-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function conversationIdFor(knowledgeBaseId) {
  if (!state.conversationIds[knowledgeBaseId]) {
    state.conversationIds[knowledgeBaseId] = createConversationId();
  }
  return state.conversationIds[knowledgeBaseId];
}

function syncConversationControls() {
  if (!state.activeKnowledgeBase) return;
  $("#conversation-id").value = conversationIdFor(state.activeKnowledgeBase.id);
}

function configuredConversationId() {
  const knowledgeBaseId = state.activeKnowledgeBase.id;
  const requested = $("#conversation-id").value.trim();
  const conversationId = (requested || createConversationId()).slice(0, 120);
  state.conversationIds[knowledgeBaseId] = conversationId;
  $("#conversation-id").value = conversationId;
  return conversationId;
}

function startNewConversation() {
  if (!state.activeKnowledgeBase) return;
  state.conversationIds[state.activeKnowledgeBase.id] = createConversationId();
  state.lastRun = null;
  syncConversationControls();
  renderConversationIntro();
  toast("已开始新会话，会话记忆已清空");
}

function exampleQuestions(description = "") {
  const marker = "示例问题：";
  const markerIndex = description.indexOf(marker);
  if (markerIndex < 0) return [];
  return description.slice(markerIndex + marker.length).split("\n")
    .map((line) => line.trim().replace(/^[-•]\s*/, ""))
    .filter(Boolean)
    .slice(0, 8);
}

function renderConversationIntro() {
  const knowledgeBase = state.activeKnowledgeBase;
  if (!knowledgeBase) return;
  const markerIndex = knowledgeBase.description.indexOf("示例问题：");
  const summary = (markerIndex >= 0 ? knowledgeBase.description.slice(0, markerIndex) : knowledgeBase.description).trim();
  const questions = exampleQuestions(knowledgeBase.description);
  const suggestions = questions.length ? `
    <section class="example-panel" aria-label="示例问题">
      <div><strong>不知道怎么开始？</strong><span>点击一个真实金标问题，先填入输入框</span></div>
      <div class="example-prompts">${questions.map((question) => `
        <button class="example-prompt" type="button" data-example-question="${escapeHtml(question)}">${escapeHtml(question)}</button>
      `).join("")}</div>
    </section>` : "";
  $("#conversation").innerHTML = `
    <div class="welcome-card">
      <div class="welcome-icon">⌁</div>
      <div>
        <h2>先检索，再回答</h2>
        <p>${escapeHtml(summary || "简单问题走单轮混合 RAG；证据不足时，Agent 至多再检索一轮。")}</p>
      </div>
    </div>${suggestions}`;
  $$(".example-prompt").forEach((button) => button.addEventListener("click", () => {
    const input = $("#question");
    input.value = button.dataset.exampleQuestion;
    input.focus();
  }));
}

function showEmpty() {
  state.activeKnowledgeBase = null;
  $("#workspace").classList.add("hidden");
  $("#empty-state").classList.remove("hidden");
  $("#workspace-title").textContent = "选择或创建知识库";
}

function updateIndexPill() {
  const pill = $("#index-pill");
  const index = state.activeKnowledgeBase?.active_index_version_id;
  pill.className = `index-pill${index ? " ready" : ""}`;
  pill.querySelector("span:last-child").textContent = index ? `活动索引 ${index.slice(0, 12)}…` : "尚无活动索引";
}

async function createKnowledgeBase(event) {
  event.preventDefault();
  if (state.creatingKnowledgeBase) return;
  const nameInput = $("#kb-name");
  const name = nameInput.value.trim();
  nameInput.setCustomValidity(name ? "" : "请输入知识库名称");
  if (!nameInput.reportValidity()) return;
  try {
    setKnowledgeBaseDialogBusy(true);
    const kb = await api("/v1/knowledge-bases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, description: $("#kb-description").value.trim() }),
    });
    $("#new-kb-dialog").close("created");
    await loadKnowledgeBases(kb.id);
    toast("知识库已创建，可以上传文档了");
    activateTab("documents");
  } catch (error) { toast(error.message, true); }
  finally { setKnowledgeBaseDialogBusy(false); }
}

function setKnowledgeBaseDialogBusy(busy) {
  state.creatingKnowledgeBase = busy;
  const form = $("#new-kb-form");
  form.setAttribute("aria-busy", String(busy));
  form.querySelectorAll("button").forEach((button) => { button.disabled = busy; });
}

function resetKnowledgeBaseDialog() {
  const form = $("#new-kb-form");
  form.reset();
  $("#kb-name").setCustomValidity("");
  setKnowledgeBaseDialogBusy(false);
}

function openKnowledgeBaseDialog() {
  const dialog = $("#new-kb-dialog");
  if (dialog.open) return;
  resetKnowledgeBaseDialog();
  dialog.showModal();
  $("#kb-name").focus();
}

function closeKnowledgeBaseDialog() {
  if (state.creatingKnowledgeBase) return;
  const dialog = $("#new-kb-dialog");
  if (dialog.open) dialog.close("cancel");
}

async function uploadFiles(files) {
  if (!state.activeKnowledgeBase || !files.length) return;
  const form = new FormData();
  [...files].forEach((file) => form.append("files", file));
  form.append("auto_index", "false");
  showJobBanner(`正在上传 ${files.length} 个文件…`);
  try {
    const response = await api(`/v1/knowledge-bases/${state.activeKnowledgeBase.id}/documents`, { method: "POST", body: form });
    response.items.forEach((item) => pollJob(item.job_id));
    toast(`${response.items.length} 个文件已进入解析队列`);
    await loadDocuments();
  } catch (error) { toast(error.message, true); showJobBanner(error.message, true); }
  finally { $("#file-input").value = ""; }
}

async function pollJob(jobId, onComplete = null) {
  if (state.polling.has(jobId)) return;
  state.polling.add(jobId);
  try {
    for (;;) {
      const job = await api(`/v1/jobs/${jobId}`);
      showJobBanner(`${job.stage} · ${Math.round(job.progress * 100)}%`);
      await loadDocuments();
      if (job.status === "succeeded") {
        showJobBanner("任务完成");
        if (onComplete) await onComplete(job);
        await refreshActiveKnowledgeBase();
        break;
      }
      if (job.status === "failed") throw new Error(job.error || "任务失败");
      await new Promise((resolve) => setTimeout(resolve, 900));
    }
  } catch (error) { showJobBanner(error.message, true); toast(error.message, true); }
  finally { state.polling.delete(jobId); }
}

function showJobBanner(message, error = false) {
  const banner = $("#job-banner");
  banner.textContent = message;
  banner.classList.remove("hidden");
  banner.style.color = error ? "#8e342d" : "";
}

async function loadDocuments() {
  if (!state.activeKnowledgeBase) return;
  const documents = await api(`/v1/knowledge-bases/${state.activeKnowledgeBase.id}/documents`);
  $("#document-list").innerHTML = documents.map((document) => `
    <div class="document-row">
      <div><strong>${escapeHtml(document.filename)}</strong><span>${formatBytes(document.size_bytes)} · ${document.id}</span></div>
      <span class="state ${document.status}">${document.status}</span>
      <span>${new Date(document.updated_at).toLocaleString()}</span>
    </div>
  `).join("") || '<div class="empty-copy">还没有文档。把文件拖到上方区域开始。</div>';
}

async function buildIndex() {
  if (!state.activeKnowledgeBase) return;
  try {
    $("#build-index-button").disabled = true;
    const result = await api(`/v1/knowledge-bases/${state.activeKnowledgeBase.id}/index-builds`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contextualize: false,
        activate: true,
        chunk_strategy: $("#chunk-strategy").value,
        dense_backend: $("#dense-backend").value,
      }),
    });
    showJobBanner("索引任务已排队");
    await pollJob(result.job_id, async () => { toast("新索引已激活"); });
  } catch (error) { toast(error.message, true); }
  finally { $("#build-index-button").disabled = false; }
}

async function refreshActiveKnowledgeBase() {
  if (!state.activeKnowledgeBase) return;
  const kb = await api(`/v1/knowledge-bases/${state.activeKnowledgeBase.id}`);
  state.activeKnowledgeBase = kb;
  state.knowledgeBases = state.knowledgeBases.map((item) => item.id === kb.id ? kb : item);
  updateIndexPill(); renderKnowledgeBases();
}

async function ask(event) {
  event.preventDefault();
  if (!state.activeKnowledgeBase) return;
  const input = $("#question");
  const question = input.value.trim();
  if (!question) return;
  appendMessage("user", question);
  input.value = "";
  $("#ask-button").disabled = true;
  const pending = appendMessage("assistant", "正在检索、重排并检查证据…", { pending: true });
  try {
    const requestedMemory = Number.parseInt($("#memory-turns").value, 10);
    const memoryTurns = Number.isFinite(requestedMemory)
      ? Math.max(0, Math.min(10, requestedMemory))
      : 5;
    $("#memory-turns").value = String(memoryTurns);
    const requestedConcurrency = Number.parseInt($("#retrieval-concurrency").value, 10);
    const retrievalConcurrency = Number.isFinite(requestedConcurrency)
      ? Math.max(1, Math.min(4, requestedConcurrency))
      : 2;
    $("#retrieval-concurrency").value = String(retrievalConcurrency);
    const response = await api("/v1/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        knowledge_base_id: state.activeKnowledgeBase.id,
        question,
        conversation_id: configuredConversationId(),
        memory_turns: memoryTurns,
        retrieval_concurrency: retrievalConcurrency,
      }),
    });
    pending.remove();
    appendMessage("assistant", response.answer, response);
    state.lastRun = await api(response.trace_url);
    renderTrace(state.lastRun);
  } catch (error) { pending.remove(); appendMessage("assistant", `请求失败：${error.message}`); }
  finally { $("#ask-button").disabled = false; }
}

function appendMessage(role, text, metadata = {}) {
  const element = document.createElement("article");
  element.className = `message ${role}`;
  const chips = metadata.route ? `
    <div class="message-meta">
      <span class="chip">${escapeHtml(metadata.route)}</span>
      <span class="chip">${metadata.rounds} 轮检索</span>
      <span class="chip">${Math.round(metadata.latency_ms)} ms</span>
    </div>` : "";
  const citations = metadata.citations?.length ? `
    <div class="citation-list">${metadata.citations.map((citation) => `
      <div class="citation"><strong>[${escapeHtml(citation.id)}] ${escapeHtml(citation.source.filename)}</strong>
      · ${citation.source.page ? `第 ${citation.source.page} 页` : escapeHtml(citation.source.section || "文档")}
      <br>${escapeHtml(citation.quote)}</div>`).join("")}</div>` : "";
  element.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>${chips}${citations}`;
  $("#conversation").appendChild(element);
  $("#conversation").scrollTop = $("#conversation").scrollHeight;
  return element;
}

function renderTrace(run) {
  $("#trace-run-id").textContent = run.id;
  $("#trace-summary").innerHTML = `
    <span class="chip">${escapeHtml(run.route || run.status)}</span>
    <span class="chip">${run.rounds} 轮</span>
    <span class="chip">${Math.round(run.metrics.latency_ms || 0)} ms</span>
    <span class="chip">${run.trace.length} 个事件</span>`;
  $("#trace-timeline").className = "timeline";
  $("#trace-timeline").innerHTML = run.trace.map((event) => `
    <article class="timeline-item">
      <h3>${escapeHtml(event.stage)}</h3>
      <p>#${event.sequence} · ${new Date(event.created_at).toLocaleTimeString()}</p>
      <details><summary>查看结构化记录</summary><pre>${escapeHtml(JSON.stringify(event.payload, null, 2))}</pre></details>
    </article>`).join("");
}

function activateTab(name) {
  $$(".tab").forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  $$(".tab-panel").forEach((panel) => panel.classList.add("hidden"));
  $(`#tab-${name}`).classList.remove("hidden");
}

function bindEvents() {
  [$("#new-kb-button"), $("#empty-new-kb")].forEach((button) => button.addEventListener("click", openKnowledgeBaseDialog));
  $("#close-kb-dialog").addEventListener("click", closeKnowledgeBaseDialog);
  $("#cancel-kb-dialog").addEventListener("click", closeKnowledgeBaseDialog);
  $("#new-kb-form").addEventListener("submit", createKnowledgeBase);
  $("#kb-name").addEventListener("input", (event) => event.target.setCustomValidity(""));
  const dialog = $("#new-kb-dialog");
  dialog.addEventListener("cancel", (event) => {
    if (state.creatingKnowledgeBase) event.preventDefault();
  });
  dialog.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    event.preventDefault();
    closeKnowledgeBaseDialog();
  });
  dialog.addEventListener("close", resetKnowledgeBaseDialog);
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) closeKnowledgeBaseDialog();
  });
  const tabs = $$(".tab");
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateTab(tab.dataset.tab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      let targetIndex = index;
      if (event.key === "ArrowLeft") targetIndex = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "ArrowRight") targetIndex = (index + 1) % tabs.length;
      if (event.key === "Home") targetIndex = 0;
      if (event.key === "End") targetIndex = tabs.length - 1;
      activateTab(tabs[targetIndex].dataset.tab);
      tabs[targetIndex].focus();
    });
  });
  $("#file-input").addEventListener("change", (event) => uploadFiles(event.target.files));
  $("#build-index-button").addEventListener("click", buildIndex);
  $("#query-form").addEventListener("submit", ask);
  $("#new-conversation-button").addEventListener("click", startNewConversation);
  const dropzone = $("#dropzone");
  ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); }));
  dropzone.addEventListener("drop", (event) => uploadFiles(event.dataTransfer.files));
  dropzone.addEventListener("keydown", (event) => {
    if (!["Enter", " "].includes(event.key)) return;
    event.preventDefault();
    $("#file-input").click();
  });
}

bindEvents();
await checkHealth();
try { await loadKnowledgeBases(); } catch (error) { toast(error.message, true); }

const BATCH_SIZE = 80;
const PREDICT_CHUNK_SIZE = 20;

const ISSUE_LABELS = [
  { key: "delivery_shipping", label: "Delivery & Shipping", color: "#f29d38" },
  { key: "redemption_activation", label: "Redemption & Activation", color: "#4f9bc9" },
  { key: "product_quality", label: "Product Quality", color: "#6bbf59" },
  { key: "customer_service", label: "Customer Service", color: "#d672be" },
  { key: "refund_return", label: "Refund & Return", color: "#e15a4f" },
  { key: "usability", label: "Usability", color: "#7a78e4" },
  { key: "value_price", label: "Value & Price", color: "#56b5a8" },
  { key: "fraud_scam", label: "Fraud & Scam", color: "#c9894f" },
  { key: "other", label: "Other", color: "#9da6b4" },
];

const VIEW_COPY = Object.freeze({
  customer: "Customer View: browse the product page and submit a new review.",
  admin: "Admin / CSKH Dashboard: inspect AI triage outputs, issue distribution, and the attention queue.",
});

const WORKFLOW_COPY = Object.freeze({
  badge: "[TRIAGE COMPLETE - RESOLUTION PENDING]",
  action: "Create Jira Ticket",
  context: "Status: Escalated to Customer Success (Ticket: CS-8892)",
});

const DATE_RANGE_OPTIONS = Object.freeze({
  today: "Today: Mar 14, 2026",
  yesterday: "Yesterday: Mar 13, 2026",
  last_7_days: "Last 7 Days",
  this_month: "This Month",
});

const TICKET_STATUS = Object.freeze({
  NEW: "new",
  IN_PROGRESS: "in_progress",
  RESOLVED: "resolved",
});

const dom = {
  adminViewBtn: document.getElementById("admin-view-btn"),
  customerViewBtn: document.getElementById("customer-view-btn"),
  viewSwitchText: document.getElementById("view-switch-text"),
  dateRangeSelect: document.getElementById("date-range-select"),
  dateRangeSummary: document.getElementById("date-range-summary"),
  dateRangeContext: document.getElementById("date-range-context"),
  headerBatchActions: document.getElementById("header-batch-actions"),
  currentPeriodNotes: document.querySelectorAll("[data-current-period]"),
  statusStrip: document.getElementById("status-strip"),
  adminToolbar: document.getElementById("admin-toolbar"),
  heroActionRow: document.querySelector(".admin-hero-side .admin-action-row"),
  mainProductImage: document.getElementById("main-product-image"),
  productThumbGrid: document.getElementById("product-thumb-grid"),
  recommendedGrid: document.getElementById("recommended-grid"),
  catalogContextCard: document.getElementById("catalog-context-card"),
  catalogContextImage: document.getElementById("catalog-context-image"),
  catalogContextName: document.getElementById("catalog-context-name"),
  catalogContextNote: document.getElementById("catalog-context-note"),
  heroBatchPill: document.getElementById("hero-batch-pill"),
  productStars: document.getElementById("product-stars"),
  productRatingValue: document.getElementById("product-rating-value"),
  productRatingCount: document.getElementById("product-rating-count"),
  leftOverallScore: document.getElementById("left-overall-score"),
  leftRatingCount: document.getElementById("left-rating-count"),
  ratingBars: document.getElementById("rating-bars"),
  distributionBars: document.getElementById("distribution-bars"),
  customerSayText: document.getElementById("customer-say-text"),
  customerSayTags: document.getElementById("customer-say-tags"),
  reviewsFeed: document.getElementById("reviews-feed"),
  runSampleFromLeftBtn: document.getElementById("run-sample-from-left"),
  writeReviewBtn: document.getElementById("write-review-btn"),
  reviewComposePanel: document.getElementById("review-compose-panel"),
  reviewComposeRating: document.getElementById("review-compose-rating"),
  reviewComposeText: document.getElementById("review-compose-text"),
  reviewComposeError: document.getElementById("review-compose-error"),
  reviewComposeSubmit: document.getElementById("review-compose-submit"),
  reviewComposeCancel: document.getElementById("review-compose-cancel"),
  promptChips: document.querySelectorAll(".prompt-chip"),
  runMessage: document.getElementById("run-message"),
  summaryPanel: document.getElementById("summary-panel"),
  distributionPanel: document.getElementById("distribution-panel"),
  metricsPanel: document.getElementById("metrics-panel"),
  triageFocusPanel: document.getElementById("triage-focus-panel"),
  triageFocusContent: document.getElementById("triage-focus-content"),
  mismatchChart: document.getElementById("mismatch-chart"),
  mismatchInsight: document.getElementById("mismatch-insight"),
  kpiGrid: document.getElementById("kpi-grid"),
  reviewsSectionTitle: document.getElementById("reviews-section-title"),
  reviewsSectionNote: document.getElementById("reviews-section-note"),
  opsLabelPie: document.getElementById("ops-label-pie"),
  opsLabelLegend: document.getElementById("ops-label-legend"),
  issueMixSnapshot: document.getElementById("issue-mix-snapshot"),
  railSnapshotGrid: document.getElementById("rail-snapshot-grid"),
  opsTotalFlags: document.getElementById("ops-total-flags"),
  opsBatchNote: document.getElementById("ops-batch-note"),
  singleReviewHint: document.getElementById("single-review-hint"),
  singleReviewContent: document.getElementById("single-review-content"),
  playbookContent: document.getElementById("playbook-content"),
};

let currentView = document.body.getAttribute("data-app-view") || "customer";
let catalogItems = [];
let datasetReviews = [];
let currentBatchRows = [];
let currentPredictions = [];
let currentStatus = null;
let currentDateRange = dom.dateRangeSelect?.value || "today";
const workflowStateByKey = {};

function setMessage(text, isError = false) {
  if (!dom.runMessage) return;
  dom.runMessage.textContent = text;
  dom.runMessage.style.color = isError ? "#a4371f" : "";
}

function setComposeError(text = "") {
  if (!dom.reviewComposeError) return;
  if (!text) {
    dom.reviewComposeError.textContent = "";
    dom.reviewComposeError.classList.add("hidden");
    return;
  }
  dom.reviewComposeError.textContent = text;
  dom.reviewComposeError.classList.remove("hidden");
}

function syncComposeButton() {
  if (!dom.writeReviewBtn || !dom.reviewComposePanel) return;
  const isOpen = !dom.reviewComposePanel.classList.contains("hidden");
  dom.writeReviewBtn.textContent = isOpen ? "📝 Hide Test Console" : "📝 Open Test Console";
}

function closeComposePanel(resetFields = false) {
  if (!dom.reviewComposePanel) return;
  dom.reviewComposePanel.classList.add("hidden");
  setComposeError("");
  syncComposeButton();
  if (!resetFields) return;
  if (dom.reviewComposeText) dom.reviewComposeText.value = "";
  if (dom.reviewComposeRating) dom.reviewComposeRating.value = "5";
}

function openComposePanel() {
  if (!dom.reviewComposePanel) return;
  dom.reviewComposePanel.classList.remove("hidden");
  setComposeError("");
  syncComposeButton();
  dom.reviewComposeText?.focus();
}

function emptyIssueFlags() {
  const flags = {};
  ISSUE_LABELS.forEach((item) => {
    flags[item.key] = 0;
  });
  return flags;
}

function normalizedRating(raw) {
  const parsed = Math.round(Number(raw));
  if (Number.isNaN(parsed)) return 3;
  return Math.max(1, Math.min(5, parsed));
}

function starText(count) {
  return `${"&#9733;".repeat(count)}${"&#9734;".repeat(Math.max(0, 5 - count))}`;
}

function starCountFromLabel(label, prob) {
  const normalized = String(label || "");
  if (normalized === "POSITIVE") return 5;
  if (normalized === "NEGATIVE") return 1;
  if (normalized === "NEEDS_ATTENTION") return 2;
  if (normalized === "UNCERTAIN") return 3;
  if (prob === null || prob === undefined || Number.isNaN(Number(prob))) return 3;
  return Math.max(1, Math.min(5, Math.round(Number(prob) * 5)));
}

function sentimentFromRating(rating) {
  const stars = normalizedRating(rating);
  if (stars >= 4) return "POSITIVE";
  if (stars === 3) return "UNCERTAIN";
  if (stars === 2) return "NEEDS_ATTENTION";
  return "NEGATIVE";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function labelPill(label) {
  const normalized = String(label || "").toLowerCase();
  return `<span class="pill ${normalized}">${label || "N/A"}</span>`;
}

function dateRangeLabel() {
  return DATE_RANGE_OPTIONS[currentDateRange] || DATE_RANGE_OPTIONS.today;
}

function renderHeroMeta() {
  if (dom.heroBatchPill) {
    const count = currentBatchRows.length;
    dom.heroBatchPill.textContent = `${count} live review${count === 1 ? "" : "s"}`;
  }
}

function renderDateContext() {
  const label = dateRangeLabel();
  const periodCopy = `(for ${label.toLowerCase()})`;

  dom.currentPeriodNotes?.forEach((node) => {
    node.textContent = periodCopy;
  });

  renderHeroMeta();
  updateReviewsSectionChrome();
}

function hasWorkflowEscalation(row) {
  const label = String(row?.classic_label || "");
  return label === "NEGATIVE" || label === "NEEDS_ATTENTION" || label === "UNCERTAIN";
}

function parseIssueKeys(summary) {
  const raw = String(summary || "").trim();
  if (!raw || raw === "-") return [];

  return raw
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => part.split(":")[0].trim())
    .filter((label) => ISSUE_LABELS.some((item) => item.key === label));
}

function normalizeIssueFlags(row) {
  const source = row?.issue_flags || {};
  const normalized = {};
  ISSUE_LABELS.forEach((item) => {
    const raw = source[item.key] ?? row?.[item.key] ?? 0;
    normalized[item.key] = Number(raw) >= 1 ? 1 : 0;
  });
  return normalized;
}

function activeIssueKeys(issueFlags) {
  return ISSUE_LABELS.filter((item) => Number(issueFlags[item.key]) >= 1).map((item) => item.key);
}

function issueSummaryFromFlags(issueFlags) {
  const active = activeIssueKeys(issueFlags);
  if (!active.length) return "-";
  return active.join(", ");
}

function buildIssueRowsFromPredictions(rows) {
  const counts = {};
  ISSUE_LABELS.forEach((item) => {
    counts[item.key] = 0;
  });

  rows.forEach((row) => {
    parseIssueKeys(row.issue_summary).forEach((label) => {
      counts[label] += 1;
    });
  });

  return ISSUE_LABELS.map((item) => ({
    label: item.key,
    count: counts[item.key] || 0,
  })).sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
}

function buildAttentionQueueRows(rows) {
  return [...rows]
    .filter((row) => hasWorkflowEscalation(row))
    .sort((left, right) => {
      const riskDelta = Number(right.risk_score || 0) - Number(left.risk_score || 0);
      if (riskDelta !== 0) return riskDelta;
      return Number(left.classic_probability || 0.5) - Number(right.classic_probability || 0.5);
    });
}

function renderOpsPieChart(issueRows) {
  if (!dom.opsLabelPie || !dom.opsLabelLegend || !dom.opsTotalFlags || !dom.opsBatchNote) return;

  const counts = {};
  ISSUE_LABELS.forEach((item) => {
    counts[item.key] = 0;
  });
  (issueRows || []).forEach((row) => {
    counts[row.label] = Number(row.count || 0);
  });

  const total = ISSUE_LABELS.reduce((sum, item) => sum + Number(counts[item.key] || 0), 0);
  dom.opsTotalFlags.textContent = String(total);
  dom.opsBatchNote.textContent = `Current batch: ${currentPredictions.length} reviews | Predicted issue hits: ${total}`;

  if (total <= 0) {
    dom.opsLabelPie.style.background = "conic-gradient(#d5d9d9 0deg 360deg)";
  } else {
    let cursor = 0;
    const parts = ISSUE_LABELS.map((item) => {
      const count = Number(counts[item.key] || 0);
      const start = (cursor / total) * 360;
      cursor += count;
      const end = (cursor / total) * 360;
      return `${item.color} ${start}deg ${end}deg`;
    });
    dom.opsLabelPie.style.background = `conic-gradient(${parts.join(",")})`;
  }

  dom.opsLabelLegend.innerHTML = ISSUE_LABELS.map((item) => {
    const count = Number(counts[item.key] || 0);
    const share = total > 0 ? Math.round((count / total) * 100) : 0;
    return `
      <div class="ops-legend-item">
        <span class="ops-legend-swatch" style="background:${item.color}"></span>
        <span class="ops-legend-name">${item.label}</span>
        <span class="ops-legend-value">${count} (${share}%)</span>
      </div>
    `;
  }).join("");

  if (dom.issueMixSnapshot) {
    const activeRows = (issueRows || []).filter((row) => Number(row.count || 0) > 0);
    const topDriver = activeRows[0];
    const flaggedReviews = currentPredictions.filter((row) => Number(row.issue_count || 0) > 0).length;
    const topDriverLabel = topDriver
      ? `${ISSUE_LABELS.find((item) => item.key === topDriver.label)?.label || topDriver.label} (${topDriver.count})`
      : "No issue spike";

    dom.issueMixSnapshot.innerHTML = `
      <article class="issue-snapshot-card">
        <span class="issue-snapshot-label">Top driver</span>
        <span class="issue-snapshot-value">${escapeHtml(topDriverLabel)}</span>
      </article>
      <article class="issue-snapshot-card">
        <span class="issue-snapshot-label">Reviews flagged</span>
        <span class="issue-snapshot-value">${flaggedReviews} of ${currentPredictions.length || 0}</span>
      </article>
      <article class="issue-snapshot-card">
        <span class="issue-snapshot-label">Labels active</span>
        <span class="issue-snapshot-value">${activeRows.length} active categories</span>
      </article>
    `;
  }
}

function renderRailSnapshot(summary = {}, issueRows = [], rows = []) {
  if (!dom.railSnapshotGrid) return;

  if (!Number(summary.total || 0)) {
    dom.railSnapshotGrid.innerHTML = `<p class="muted">Waiting for batch analysis...</p>`;
    return;
  }

  const activeRows = (issueRows || []).filter((row) => Number(row.count || 0) > 0);
  const topIssue = activeRows[0];
  const topIssueLabel = topIssue
    ? (ISSUE_LABELS.find((item) => item.key === topIssue.label)?.label || topIssue.label)
    : "No spike";
  const mismatch = buildMismatchSummary(Array.isArray(rows) ? rows : []);
  const hiddenRiskCopy = mismatch.high_star.total > 0
    ? `${mismatch.hiddenRisk} of ${mismatch.high_star.total}`
    : "Not available";

  const items = [
    ["Live reviews", `${summary.total}`],
    ["Flagged", `${summary.flagged || 0} cases`],
    ["Top issue", topIssueLabel],
    ["Hidden risk", hiddenRiskCopy],
  ];

  dom.railSnapshotGrid.innerHTML = items
    .map(
      ([label, value]) => `
        <article class="rail-snapshot-item">
          <span class="rail-snapshot-label">${escapeHtml(label)}</span>
          <span class="rail-snapshot-value">${escapeHtml(value)}</span>
        </article>
      `
    )
    .join("");
}

function renderTriageFocus(summary = {}, issueRows = [], rows = []) {
  if (!dom.triageFocusContent) return;

  if (!Number(summary.total || 0) || !Array.isArray(rows) || !rows.length) {
    dom.triageFocusContent.innerHTML = `<p class="muted">Focus guidance will appear after the current batch is analyzed.</p>`;
    return;
  }

  const queueRows = buildAttentionQueueRows(rows);
  const leadRow = queueRows[0] || rows[0];
  const leadPlaybook = playbookForRow(leadRow);
  const topIssue = (issueRows || []).filter((row) => Number(row.count || 0) > 0)[0];
  const topIssueLabel = topIssue
    ? (ISSUE_LABELS.find((item) => item.key === topIssue.label)?.label || topIssue.label)
    : "No active spike";
  const mismatch = buildMismatchSummary(rows);
  const hiddenShare = mismatch.high_star.total > 0
    ? `${mismatch.hiddenRisk}/${mismatch.high_star.total} high-star`
    : "No hidden-risk sample";

  const cards = [
    ["Top issue", topIssueLabel, topIssue ? `${topIssue.count} issue hits in batch` : "No issue concentration detected"],
    ["Queue pressure", `${queueRows.length} cases`, "Items currently prioritized for manual CSKH triage"],
    ["Recommended owner", leadPlaybook.owner, leadPlaybook.sla],
    ["Hidden risk", hiddenShare, "High-star reviews that still carry NLP risk signals"],
  ];

  dom.triageFocusContent.innerHTML = `
    <div class="triage-focus-grid">
      ${cards.map(([label, value, note]) => `
        <article class="triage-focus-card">
          <span class="triage-focus-label">${escapeHtml(label)}</span>
          <span class="triage-focus-value">${escapeHtml(value)}</span>
          <span class="triage-focus-note">${escapeHtml(note)}</span>
        </article>
      `).join("")}
    </div>
    <div class="triage-focus-actions">
      <p class="triage-focus-actions-title">Recommended next actions</p>
      <p class="triage-focus-note">${escapeHtml(leadPlaybook.outcome)}</p>
      <ul>
        ${leadPlaybook.checklist.slice(0, 3).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </ul>
    </div>
  `;
}

function starSummaryFromDataset(rows) {
  if (!rows.length) {
    return {
      avg: 0,
      count: 0,
      starCounts: { 5: 0, 4: 0, 3: 0, 2: 0, 1: 0 },
    };
  }

  const starCounts = { 5: 0, 4: 0, 3: 0, 2: 0, 1: 0 };
  let totalStars = 0;
  rows.forEach((row) => {
    const stars = normalizedRating(row.rating);
    totalStars += stars;
    starCounts[String(stars)] += 1;
  });

  return {
    avg: totalStars / rows.length,
    count: rows.length,
    starCounts,
  };
}

function renderProductRating(summary) {
  if (!dom.productStars || !dom.productRatingValue || !dom.productRatingCount) return;
  if (!dom.leftOverallScore || !dom.leftRatingCount) return;

  if (!summary || summary.count === 0) {
    dom.productStars.innerHTML = starText(5);
    dom.productRatingValue.textContent = "4.7 out of 5";
    dom.productRatingCount.textContent = "(0 reviews)";
    dom.leftOverallScore.textContent = "4.7";
    dom.leftRatingCount.textContent = "0";
    return;
  }

  const rounded = Math.max(1, Math.min(5, Math.round(summary.avg)));
  dom.productStars.innerHTML = starText(rounded);
  dom.productRatingValue.textContent = `${summary.avg.toFixed(1)} out of 5`;
  dom.productRatingCount.textContent = `(${summary.count} reviews)`;
  dom.leftOverallScore.textContent = summary.avg.toFixed(1);
  dom.leftRatingCount.textContent = String(summary.count);
}

function renderRatingBars(summary) {
  if (!dom.ratingBars || !dom.distributionBars) return;

  const data = summary || { count: 0, starCounts: { 5: 0, 4: 0, 3: 0, 2: 0, 1: 0 } };
  const total = data.count || 1;
  const ordered = [5, 4, 3, 2, 1];
  const html = ordered
    .map((score) => {
      const count = data.starCounts[String(score)] || 0;
      const percent = data.count ? Math.round((count / total) * 100) : 0;
      return `
      <div class="bar-row">
        <div class="bar-label">${score} star</div>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, percent)}%"></div></div>
        <div class="bar-value">${percent}%</div>
      </div>
    `;
    })
    .join("");

  dom.ratingBars.innerHTML = html;
  dom.distributionBars.innerHTML = html;
}

function refreshRatingPanels() {
  const summary = starSummaryFromDataset(currentBatchRows);
  renderHeroMeta();
  renderProductRating(summary);
  renderRatingBars(summary);
}

function relocateAdminToolbar() {
  const target = dom.headerBatchActions || dom.adminToolbar;
  if (!target || !dom.heroActionRow) return;
  if (target.contains(dom.heroActionRow)) return;
  target.appendChild(dom.heroActionRow);
  if (dom.headerBatchActions && dom.adminToolbar) {
    dom.adminToolbar.classList.add("hidden");
  }
}

function normalizeAdminLayout() {
  const catalogRail = document.querySelector(".catalog-rail");
  if (catalogRail && dom.catalogContextCard && !catalogRail.contains(dom.catalogContextCard)) {
    catalogRail.appendChild(dom.catalogContextCard);
  }

  const queueGrid = document.querySelector(".admin-queue-grid");
  const sideRail = queueGrid?.querySelector(".queue-side-rail");
  const attentionQueue = document.getElementById("attention-queue");
  const manualTest = document.getElementById("manual-test");
  if (!queueGrid || !sideRail || !attentionQueue || !manualTest) return;

  let queueMainColumn = queueGrid.querySelector(".queue-main-column");
  if (!queueMainColumn) {
    queueMainColumn = document.createElement("div");
    queueMainColumn.className = "queue-main-column";
    queueGrid.insertBefore(queueMainColumn, sideRail);
  }

  if (!queueMainColumn.contains(attentionQueue)) {
    queueMainColumn.appendChild(attentionQueue);
  }

  if (!queueMainColumn.contains(manualTest)) {
    queueMainColumn.appendChild(manualTest);
  }
}

function formatModelDate(raw) {
  const value = String(raw || "").trim();
  if (!value || value === "N/A") return "Build date unavailable";

  const normalized = value.includes("T") ? value : value.replace(" ", "T");
  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) return value;

  return parsed.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function renderStatus(status) {
  dom.statusStrip?.remove();
  return;
  const classic = status?.classic || {};
  const transformer = status?.transformer || {};
  const modelInfo = classic.model_info || {};
  const classicLoaded = Boolean(classic.loaded);
  const transformerLoaded = Boolean(transformer.loaded);
  const issueMode = classic.issue_mode || "Issue routing unavailable";
  const buildMeta = [
    modelInfo.k_features ? `K* ${modelInfo.k_features}` : "",
    modelInfo.thresholds ? `thresholds ${modelInfo.thresholds}` : "",
  ]
    .filter(Boolean)
    .join(" · ");
  const buildValue = [
    modelInfo.variant ? `Variant ${modelInfo.variant}` : "Classic bundle",
    formatModelDate(modelInfo.trained_at),
  ]
    .filter(Boolean)
    .join(" · ");
  const cards = [
    {
      icon: "🚦",
      label: "Live Mode",
      value: classicLoaded ? "Production runtime online" : "Runtime unavailable",
      meta: classicLoaded
        ? "Ready for batch triage and review-level predictions."
        : (classic.message || "Classic runtime is offline."),
      tone: classicLoaded ? "production" : "",
    },
    {
      icon: "🧠",
      label: "Active Pipeline",
      value: classicLoaded ? "Classic ML + trained issue classifier" : "Inference offline",
      meta: transformerLoaded
        ? "Transformer is active for full-text context."
        : `${issueMode}. Transformer remains optional and is currently disabled.`,
      tone: classicLoaded ? "inference" : "",
    },
    {
      icon: transformerLoaded ? "🤖" : "🧾",
      label: transformerLoaded ? "Deep Model" : "Model Build",
      value: transformerLoaded ? "Transformer active" : buildValue,
      meta: transformerLoaded
        ? (transformer.message || "Full-text context is enabled for this runtime.")
        : (buildMeta || "Build metadata unavailable."),
      tone: transformerLoaded ? "transformer" : "",
    },
  ];

  dom.statusStrip.innerHTML = cards
    .map(
      ({ icon, label, value, meta, tone }) => `
      <article class="status-card${tone ? ` status-card--${tone}` : ""}">
        <div class="status-card-head">
          <span class="status-icon" aria-hidden="true">${icon || "•"}</span>
          <div class="status-copy">
            <p class="status-label">${label}</p>
            <p class="status-value${tone ? ` status-value--${tone}` : ""}">${escapeHtml(value)}</p>
          </div>
        </div>
        <p class="status-meta">${escapeHtml(meta || "")}</p>
      </article>
    `
    )
    .join("");
}

function niceNameFromFile(fileName, index) {
  const base = String(fileName || "").replace(/\.[^.]+$/, "");
  if (!base) return `Recommended Item ${index + 1}`;
  const short = base.slice(0, 26);
  return `Gift Card Item ${index + 1} - ${short}`;
}

function itemDisplayName(item, index) {
  const value = String(item?.display_name || "").trim();
  if (value) return value;
  return niceNameFromFile(item?.name, index);
}

function itemSubtitle(item) {
  const value = String(item?.subtitle || "").trim();
  if (value) return value;
  return "Prime eligible | Free returns";
}

function itemBadge(item) {
  const value = String(item?.badge || "").trim();
  if (value) return value;
  return "Prime";
}

function itemPriceVnd(item, index) {
  const parsed = Number(item?.price_vnd);
  if (Number.isFinite(parsed) && parsed > 0) return Math.round(parsed);
  return 349000 + index * 37000;
}

function itemRating(item, index) {
  const parsed = Number(item?.rating);
  if (Number.isFinite(parsed) && parsed > 0) return Math.max(1, Math.min(5, parsed));
  return Math.max(3.8, Math.min(4.9, 4.2 + (index % 4) * 0.15));
}

function renderCatalogContext(items) {
  if (!dom.catalogContextCard || !dom.catalogContextImage || !dom.catalogContextName || !dom.catalogContextNote) {
    return;
  }

  const firstItem = Array.isArray(items) && items.length ? items[0] : null;
  if (!firstItem?.url) {
    dom.catalogContextCard.classList.add("hidden");
    return;
  }

  dom.catalogContextCard.classList.remove("hidden");
  dom.catalogContextImage.src = firstItem.url;
  dom.catalogContextImage.alt = itemDisplayName(firstItem, 0);
  dom.catalogContextName.textContent = itemDisplayName(firstItem, 0);
  dom.catalogContextNote.textContent = "Reference visual for the current review set.";
}

function renderRecommendedProducts(items) {
  if (!dom.recommendedGrid) return;
  if (!items.length) {
    dom.recommendedGrid.innerHTML = `<p class="muted">No item images found in /items.</p>`;
    return;
  }

  dom.recommendedGrid.innerHTML = items
    .slice(0, 10)
    .map((item, idx) => {
      const price = itemPriceVnd(item, idx).toLocaleString("vi-VN");
      const rating = itemRating(item, idx);
      return `
      <article class="recommended-card">
        <p class="recommended-badge">${escapeHtml(itemBadge(item))}</p>
        <img src="${item.url}" alt="${item.name}" />
        <p class="recommended-title">${escapeHtml(itemDisplayName(item, idx))}</p>
        <p class="recommended-meta">${escapeHtml(itemSubtitle(item))}</p>
        <p class="recommended-rating">${starText(Math.round(rating))} ${rating.toFixed(1)}</p>
        <p class="recommended-price">VND ${price}</p>
      </article>
      `;
    })
    .join("");
}

function renderProductGallery(items) {
  if (!dom.mainProductImage || !dom.productThumbGrid) return;

  if (!items.length) {
    dom.mainProductImage.src =
      "data:image/svg+xml;charset=UTF-8," +
      encodeURIComponent(
        '<svg xmlns="http://www.w3.org/2000/svg" width="480" height="720"><rect width="100%" height="100%" fill="#f2f5f6"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#52666f" font-size="22" font-family="Segoe UI">No images in /items</text></svg>'
      );
    dom.productThumbGrid.innerHTML = "";
    return;
  }

  dom.mainProductImage.src = items[0].url;
  dom.mainProductImage.alt = items[0].name;

  dom.productThumbGrid.innerHTML = items
    .slice(0, 8)
    .map(
      (item, idx) => `
      <button class="thumb ${idx === 0 ? "active" : ""}" type="button" data-url="${item.url}" data-name="${item.name}">
        <img src="${item.url}" alt="${item.name}" />
      </button>
    `
    )
    .join("");

  const thumbs = dom.productThumbGrid.querySelectorAll(".thumb");
  thumbs.forEach((thumb) => {
    thumb.addEventListener("click", () => {
      const url = thumb.getAttribute("data-url");
      const name = thumb.getAttribute("data-name");
      if (!url) return;
      dom.mainProductImage.src = url;
      dom.mainProductImage.alt = name || "Product image";
      thumbs.forEach((node) => node.classList.remove("active"));
      thumb.classList.add("active");
    });
  });
}

function reviewHeadline(label) {
  if (label === "POSITIVE") return "Great overall experience";
  if (label === "NEGATIVE") return "Needs immediate attention";
  if (label === "NEEDS_ATTENTION") return "Good but needs follow-up";
  return "Uncertain sentiment, manual check needed";
}

function queueText(text) {
  const raw = String(text || "").trim();
  if (raw === "Product Quality hears new item") {
    return "Product is okay, but user manual is missing.";
  }
  return raw;
}

function queueSentimentMeta(label) {
  if (label === "NEGATIVE") {
    return { text: "Negative", tone: "negative" };
  }
  if (label === "NEEDS_ATTENTION") {
    return { text: "Mixed", tone: "warning" };
  }
  if (label === "UNCERTAIN") {
    return { text: "Manual Review", tone: "neutral" };
  }
  return { text: "Positive", tone: "positive" };
}

function normalizedRiskScore(rawScore) {
  const parsed = Number(rawScore);
  if (!Number.isFinite(parsed)) return 0;
  return Math.max(0, Math.min(100, Math.round(parsed / 4.2)));
}

function riskPriorityMeta(rawScore) {
  const score = normalizedRiskScore(rawScore);
  if (score >= 85) return { text: "Critical", tone: "critical" };
  if (score >= 60) return { text: "High", tone: "high" };
  return { text: "Monitor", tone: "monitor" };
}

function playbookForRow(row) {
  const issueLabels = parseIssueKeys(row?.issue_summary);
  const primary = issueLabels[0] || "other";
  const risk = normalizedRiskScore(row?.risk_score);

  const defaults = {
    owner: "CSKH Ops",
    sla: risk >= 85 ? "Respond in 2h" : risk >= 60 ? "Respond in 8h" : "Monitor in 24h",
    outcome: "Acknowledge issue and confirm next step with the customer.",
    checklist: [
      "Review the full customer narrative and confirm issue scope.",
      "Validate whether a ticket already exists before creating a new handoff.",
      "Send the customer a clear status update with next action and expected timeline.",
    ],
  };

  const overrides = {
    delivery_shipping: {
      owner: "Fulfillment + CSKH",
      outcome: "Offer recovery for delayed delivery and confirm shipment path.",
      checklist: [
        "Check carrier and fulfillment event logs for delay or failed handoff.",
        "Offer resend or recovery credit when SLA breach is confirmed.",
        "Escalate repeat delivery failures to logistics operations.",
      ],
    },
    redemption_activation: {
      owner: "Gift Card Platform Ops",
      outcome: "Verify code state and restore customer usability quickly.",
      checklist: [
        "Check activation and redemption logs for the card code.",
        "Reissue or reactivate only after status is verified in the platform.",
        "Escalate platform mismatch cases to gift-card operations.",
      ],
    },
    refund_return: {
      owner: "Refund Desk",
      outcome: "Decide refund or replacement path based on usage status.",
      checklist: [
        "Verify usage history and policy eligibility for refund.",
        "Offer replacement or refund exception if the card is unusable.",
        "Document the resolution path for audit and policy tuning.",
      ],
    },
    customer_service: {
      owner: "CSKH Supervisor",
      outcome: "Recover trust and close the service failure loop.",
      checklist: [
        "Review prior agent interactions and identify breakdown point.",
        "Route for supervisor callback when multiple contacts already failed.",
        "Capture the case for coaching or QA feedback if needed.",
      ],
    },
    fraud_scam: {
      owner: "Trust & Safety",
      outcome: "Prevent loss and validate the claim before refunding.",
      checklist: [
        "Lock or review suspicious balance/redemption patterns immediately.",
        "Validate purchase, code history, and potential abuse signals.",
        "Avoid auto-refund until trust-and-safety review is complete.",
      ],
    },
    product_quality: {
      owner: "Catalog / Vendor Ops",
      outcome: "Resolve product defect or packaging issue with evidence.",
      checklist: [
        "Capture defect details from the review and prior similar complaints.",
        "Offer replacement or refund path if product quality is confirmed.",
        "Escalate repeated quality spikes to vendor or catalog operations.",
      ],
    },
    value_price: {
      owner: "Commercial Ops",
      outcome: "Address perceived value gap and check pricing consistency.",
      checklist: [
        "Review current pricing, promotion history, and expectation mismatch.",
        "Check whether packaging or value proposition caused the complaint.",
        "Feed recurring value complaints back to merchandizing decisions.",
      ],
    },
  };

  const chosen = { ...defaults, ...(overrides[primary] || {}) };
  return {
    owner: chosen.owner,
    sla: chosen.sla,
    outcome: chosen.outcome,
    checklist: chosen.checklist,
  };
}

function reviewKey(row, index = 0) {
  const direct = String(row?.review_key || row?.id || "").trim();
  if (direct) return direct;
  return `${String(row?.text || "").trim().slice(0, 120)}::${index}`;
}

function deriveInitialWorkflowState(row, index = 0) {
  const score = normalizedRiskScore(row?.risk_score);
  if (score >= 80) {
    return { status: TICKET_STATUS.NEW, ticketId: null };
  }
  if (score >= 55) {
    return { status: TICKET_STATUS.IN_PROGRESS, ticketId: `ZD-${8800 + index}` };
  }
  return { status: TICKET_STATUS.RESOLVED, ticketId: `ZD-${8600 + index}` };
}

function ensureWorkflowState(row, index = 0) {
  const key = reviewKey(row, index);
  if (!workflowStateByKey[key]) {
    workflowStateByKey[key] = deriveInitialWorkflowState(row, index);
  }
  return workflowStateByKey[key];
}

function ticketStatusMeta(status) {
  if (status === TICKET_STATUS.IN_PROGRESS) {
    return { text: "In Progress", tone: "in-progress" };
  }
  if (status === TICKET_STATUS.RESOLVED) {
    return { text: "Resolved", tone: "resolved" };
  }
  return { text: "New", tone: "new" };
}

function ticketActionMeta(state) {
  if (state.status === TICKET_STATUS.IN_PROGRESS) {
    return {
      action: "view",
      label: "View Zendesk Ticket",
      tone: "secondary",
      disabled: false,
    };
  }
  if (state.status === TICKET_STATUS.RESOLVED) {
    return {
      action: "closed",
      label: "Closed - No Action",
      tone: "disabled",
      disabled: true,
    };
  }
  return {
    action: "create",
    label: "Create Ticket",
    tone: "primary",
    disabled: false,
  };
}

function attachSourceMetadata(sourceRows, predictions) {
  return predictions.map((prediction, index) => ({
    ...prediction,
    review_key: sourceRows[index]?.id || reviewKey(sourceRows[index], index),
    source_rating: normalizedRating(sourceRows[index]?.rating ?? 3),
  }));
}

function issueTagMarkup(issueSummary, limit = null) {
  const labels = parseIssueKeys(issueSummary);
  if (!labels.length) {
    return `<span class="queue-issue-tag queue-issue-tag--muted">No issue labels</span>`;
  }

  const visible = limit ? labels.slice(0, limit) : labels;
  const html = visible
    .map((label) => {
      const match = ISSUE_LABELS.find((item) => item.key === label);
      return `<span class="queue-issue-tag">${escapeHtml(match?.label || label)}</span>`;
    })
    .join("");

  if (!limit || labels.length <= limit) return html;
  return `${html}<span class="queue-issue-tag queue-issue-tag--muted">+${labels.length - limit} more</span>`;
}

function renderSingleReviewAnalysis(row, displayIndex = 0) {
  if (!dom.singleReviewContent || !dom.singleReviewHint || !dom.playbookContent) return;

  if (!row || currentView !== "admin") {
    dom.singleReviewHint.textContent =
      "Click Analyze on a queue item to inspect prediction details for that review.";
    dom.singleReviewContent.innerHTML = "";
    dom.playbookContent.innerHTML = `<p class="muted">Select a queue item to load the recommended recovery path.</p>`;
    return;
  }

  dom.singleReviewHint.textContent = `Showing analysis for queue item #${displayIndex}.`;
  const workflow = ensureWorkflowState(row, displayIndex - 1);
  const workflowMeta = ticketStatusMeta(workflow.status);
  const normalizedRisk = normalizedRiskScore(row.risk_score);
  const priorityMeta = riskPriorityMeta(row.risk_score);
  const issueLabels = parseIssueKeys(row.issue_summary);
  const primaryDriver = issueLabels.length
    ? (ISSUE_LABELS.find((item) => item.key === issueLabels[0])?.label || issueLabels[0])
    : "None";
  const playbook = playbookForRow(row);
  const workflowHtml = hasWorkflowEscalation(row)
    ? `
    <div class="workflow-block">
      <span class="workflow-badge workflow-badge--${workflowMeta.tone}">${workflowMeta.text}</span>
      <p class="workflow-meta">${WORKFLOW_COPY.badge}</p>
      <p class="workflow-meta">
        ${workflow.ticketId ? `Zendesk ticket: ${escapeHtml(workflow.ticketId)}` : "Zendesk ticket not created yet."}
      </p>
      <p class="workflow-meta">Current next action: ${escapeHtml(ticketActionMeta(workflow).label)}</p>
    </div>
  `
    : "";
  dom.singleReviewContent.innerHTML = `
    <div class="single-review-summary">
      <article class="single-review-summary-card">
        <span class="single-review-summary-label">Priority</span>
        <span class="single-review-summary-value">${escapeHtml(priorityMeta.text)}</span>
      </article>
      <article class="single-review-summary-card">
        <span class="single-review-summary-label">Source Rating</span>
        <span class="single-review-summary-value">${escapeHtml(String(row.source_rating ?? 3))}/5</span>
      </article>
      <article class="single-review-summary-card">
        <span class="single-review-summary-label">Primary Driver</span>
        <span class="single-review-summary-value">${escapeHtml(primaryDriver)}</span>
      </article>
      <article class="single-review-summary-card">
        <span class="single-review-summary-label">Issue Labels</span>
        <span class="single-review-summary-value">${issueLabels.length}</span>
      </article>
    </div>
    <table class="single-review-table">
      <tbody>
        <tr><th>Label</th><td>${labelPill(row.classic_label)}</td></tr>
        <tr><th>Queue Tag</th><td>${escapeHtml(queueSentimentMeta(row.classic_label).text)}</td></tr>
        <tr><th>Probability</th><td>${row.classic_probability == null ? "N/A" : Number(row.classic_probability).toFixed(3)}</td></tr>
        <tr><th>Risk Score</th><td>${normalizedRisk}/100 <span class="muted">(raw ${escapeHtml(String(row.risk_score ?? "N/A"))})</span></td></tr>
        <tr><th>Ticket Status</th><td>${escapeHtml(workflowMeta.text)}</td></tr>
        <tr><th>Issue Labels</th><td>${escapeHtml(row.issue_summary || "-")}</td></tr>
        <tr><th>Review</th><td class="single-review-text-cell">${escapeHtml(queueText(row.text || ""))}</td></tr>
      </tbody>
    </table>
    ${workflowHtml}
  `;

  dom.playbookContent.innerHTML = `
    <div class="playbook-kpis">
      <article class="playbook-card">
        <span class="playbook-card-label">Owner</span>
        <span class="playbook-card-value">${escapeHtml(playbook.owner)}</span>
      </article>
      <article class="playbook-card">
        <span class="playbook-card-label">Target SLA</span>
        <span class="playbook-card-value">${escapeHtml(playbook.sla)}</span>
      </article>
      <article class="playbook-card">
        <span class="playbook-card-label">Primary Driver</span>
        <span class="playbook-card-value">${escapeHtml(primaryDriver)}</span>
      </article>
      <article class="playbook-card">
        <span class="playbook-card-label">Customer Outcome</span>
        <span class="playbook-card-value">${escapeHtml(playbook.outcome)}</span>
      </article>
    </div>
    <ol class="playbook-list">
      ${playbook.checklist.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
    </ol>
  `;
}

function renderBatchReviewFeed(rows) {
  if (!dom.reviewsFeed) return;
  if (!rows.length) {
    dom.reviewsFeed.innerHTML = currentView === "admin"
      ? `<p class="muted">No rows currently require manual attention.</p>`
      : `<p class="muted">No customer reviews available.</p>`;
    renderSingleReviewAnalysis(null);
    return;
  }

  const displayedRows = currentView === "admin"
    ? buildAttentionQueueRows(rows).slice(0, 20)
    : rows.slice(0, 12);
  if (currentView !== "admin") {
    dom.reviewsFeed.innerHTML = displayedRows
      .map((row, idx) => {
        const stars = starCountFromLabel(row.classic_label, row.classic_probability);
        return `
        <article class="review-card public-review-card">
          <div class="review-head-main">
            <div class="avatar">${idx + 1}</div>
            <div>
              <p class="review-title">${reviewHeadline(row.classic_label)}</p>
              <p class="review-meta">Verified customer review | ${starText(stars)}</p>
            </div>
          </div>
          <p class="review-text">${escapeHtml(queueText(row.text || ""))}</p>
        </article>
        `;
      })
      .join("");
    return;
  }

  dom.reviewsFeed.innerHTML = `
    <div class="table-wrap queue-table-wrap">
      <table class="queue-table">
        <colgroup>
          <col style="width:88px">
          <col style="width:52%">
          <col style="width:22%">
          <col style="width:14%">
        </colgroup>
        <thead>
          <tr>
            <th>Queue</th>
            <th>Review & Signals</th>
            <th>Risk / Status</th>
            <th>Take Action</th>
          </tr>
        </thead>
        <tbody>
          ${displayedRows
            .map((row, idx) => {
              const sentiment = queueSentimentMeta(row.classic_label);
              const workflow = ensureWorkflowState(row, idx);
              const statusMeta = ticketStatusMeta(workflow.status);
              const actionMeta = ticketActionMeta(workflow);
              const risk = normalizedRiskScore(row.risk_score);
              const priorityMeta = riskPriorityMeta(row.risk_score);
              const issueLabels = parseIssueKeys(row.issue_summary);
              const primaryDriver = issueLabels.length
                ? (ISSUE_LABELS.find((item) => item.key === issueLabels[0])?.label || issueLabels[0])
                : "none";
              return `
              <tr class="queue-table-row" data-review-index="${idx}">
                <td>
                  <div class="queue-item-cell">
                    <span class="queue-item-id">#${idx + 1}</span>
                    <button type="button" class="queue-inspect-btn" data-review-index="${idx}">Inspect</button>
                  </div>
                </td>
                <td>
                  <p class="queue-review-snippet">${escapeHtml(queueText(row.text || ""))}</p>
                  <div class="queue-signal-line">
                    <span class="queue-sentiment-badge queue-sentiment-badge--${sentiment.tone}">
                      ${escapeHtml(sentiment.text)}
                    </span>
                    <span class="queue-meta-line">Source rating: ${escapeHtml(String(row.source_rating ?? 3))}/5</span>
                    <span class="queue-meta-line">Confidence: ${escapeHtml(row.classic_confidence || "N/A")}</span>
                  </div>
                  <div class="queue-issue-tags">${issueTagMarkup(row.issue_summary, 4)}</div>
                  <div class="queue-cell-meta">
                    <p class="queue-meta-line">Queue tag: ${escapeHtml(row.classic_label || "N/A")}</p>
                    <p class="queue-meta-line">Primary driver: ${escapeHtml(primaryDriver)}</p>
                    <p class="queue-meta-line">${issueLabels.length} active label${issueLabels.length === 1 ? "" : "s"}</p>
                  </div>
                </td>
                <td>
                  <div class="queue-status-stack">
                    <div>
                      <span class="queue-risk-value">${risk}</span>
                      <span class="queue-risk-meta">/ 100</span>
                    </div>
                    <div class="queue-risk-bar" aria-hidden="true">
                      <span class="queue-risk-bar-fill" style="width:${risk}%"></span>
                    </div>
                    <span class="queue-priority queue-priority--${priorityMeta.tone}">${escapeHtml(priorityMeta.text)}</span>
                    <span class="queue-status-badge queue-status-badge--${statusMeta.tone}">
                      ${escapeHtml(statusMeta.text)}
                    </span>
                    <p class="queue-meta-line">${workflow.ticketId ? escapeHtml(workflow.ticketId) : "No ticket yet"}</p>
                    <p class="queue-meta-line">Owner: CSKH Ops</p>
                  </div>
                </td>
                <td>
                  <div class="queue-action-cell">
                    <button
                      type="button"
                      class="queue-action-btn queue-action-btn--${actionMeta.tone}"
                      data-review-index="${idx}"
                      data-action="${actionMeta.action}"
                      ${actionMeta.disabled ? "disabled" : ""}
                    >
                      ${escapeHtml(actionMeta.label)}
                    </button>
                    <p class="queue-meta-line">${escapeHtml(
                      actionMeta.action === "create"
                        ? "Route into Zendesk handoff."
                        : actionMeta.action === "view"
                          ? "Open the active ticket thread."
                          : "No further action required."
                    )}</p>
                  </div>
                </td>
              </tr>
              `;
            })
            .join("")}
        </tbody>
      </table>
    </div>
  `;

  const inspectButtons = dom.reviewsFeed.querySelectorAll(".queue-inspect-btn");
  inspectButtons.forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      const idx = Number(button.getAttribute("data-review-index"));
      renderSingleReviewAnalysis(displayedRows[idx], idx + 1);
    });
  });

  const queueRows = dom.reviewsFeed.querySelectorAll(".queue-table-row");
  queueRows.forEach((rowNode) => {
    rowNode.addEventListener("click", (event) => {
      if (event.target.closest("button")) return;
      const idx = Number(rowNode.getAttribute("data-review-index"));
      renderSingleReviewAnalysis(displayedRows[idx], idx + 1);
    });
  });

  const actionButtons = dom.reviewsFeed.querySelectorAll(".queue-action-btn");
  actionButtons.forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      const idx = Number(button.getAttribute("data-review-index"));
      const action = String(button.getAttribute("data-action") || "");
      const row = displayedRows[idx];
      const key = reviewKey(row, idx);
      const state = ensureWorkflowState(row, idx);

      if (action === "create") {
        const ticketId = `ZD-${9000 + idx}`;
        workflowStateByKey[key] = {
          status: TICKET_STATUS.IN_PROGRESS,
          ticketId,
        };
        renderBatchReviewFeed(currentPredictions);
        renderSingleReviewAnalysis(row, idx + 1);
        setMessage(`Ticket ${ticketId} created for queue item #${idx + 1}.`);
        return;
      }

      if (action === "view") {
        renderSingleReviewAnalysis(row, idx + 1);
        setMessage(`Mocked handoff: opened Zendesk ticket ${state.ticketId}.`);
      }
    });
  });
}

function buildFallbackPredictions(rows) {
  return rows.map((row, index) => {
    const rating = normalizedRating(row.rating);
    const label = sentimentFromRating(rating);
    const flags = normalizeIssueFlags(row);
    const issueCount = activeIssueKeys(flags).length;
    return {
      review_key: row.id || reviewKey(row, index),
      text: row.text || "",
      classic_label: label,
      classic_probability: rating / 5,
      classic_confidence: "Dataset",
      fallback_reason: "dataset_seed",
      issue_summary: issueSummaryFromFlags(flags),
      issue_count: issueCount,
      risk_score: Math.max(0, (6 - rating) * 70 + issueCount * 18),
      transformer_label: null,
      transformer_probability: null,
      transformer_confidence: null,
      transformer_reason: null,
      agreement: null,
    };
  });
}

function buildSummaryFromPredictions(rows) {
  const summary = {
    total: rows.length,
    flagged: 0,
    negative: 0,
    needs_attention: 0,
    uncertain: 0,
    positive: 0,
  };

  rows.forEach((row) => {
    const label = String(row.classic_label || "");
    if (label === "NEGATIVE") summary.negative += 1;
    else if (label === "NEEDS_ATTENTION") summary.needs_attention += 1;
    else if (label === "UNCERTAIN") summary.uncertain += 1;
    else if (label === "POSITIVE") summary.positive += 1;
  });

  summary.flagged = summary.negative + summary.needs_attention;
  return summary;
}

function hasNlpRisk(row) {
  const label = String(row?.classic_label || "");
  return label !== "POSITIVE" || Number(row?.issue_count || 0) > 0;
}

function buildMismatchSummary(rows) {
  const summary = {
    low_star: { label: "1-3 star reviews", total: 0, risk: 0, clean: 0 },
    high_star: { label: "4-5 star reviews", total: 0, risk: 0, clean: 0 },
    hiddenRisk: 0,
  };

  rows.forEach((row) => {
    const stars = normalizedRating(row?.source_rating ?? row?.rating ?? 3);
    const bucket = stars >= 4 ? summary.high_star : summary.low_star;
    bucket.total += 1;

    if (hasNlpRisk(row)) {
      bucket.risk += 1;
      if (stars >= 4) summary.hiddenRisk += 1;
    } else {
      bucket.clean += 1;
    }
  });

  return summary;
}

function mismatchRowMarkup(bucket, riskLabel) {
  const total = Math.max(bucket.total, 1);
  const riskShare = bucket.total ? (bucket.risk / total) * 100 : 0;
  const cleanShare = bucket.total ? (bucket.clean / total) * 100 : 0;

  return `
    <div class="mismatch-row">
      <div class="mismatch-row-head">
        <p class="mismatch-row-label">${bucket.label}</p>
        <p class="mismatch-row-meta">
          ${bucket.total} reviews | ${bucket.risk} ${riskLabel} | ${bucket.clean} aligned
        </p>
      </div>
      <div class="mismatch-bar" role="img" aria-label="${bucket.label}">
        <span class="mismatch-segment mismatch-segment--risk" style="width:${riskShare}%"></span>
        <span class="mismatch-segment mismatch-segment--clean" style="width:${cleanShare}%"></span>
      </div>
    </div>
  `;
}

function renderMismatchChart(rows) {
  if (!dom.mismatchChart || !dom.mismatchInsight) return;

  if (!rows.length) {
    dom.mismatchChart.innerHTML = "";
    dom.mismatchInsight.textContent =
      "Mismatch insight will appear after the current batch is analyzed.";
    return;
  }

  const summary = buildMismatchSummary(rows);
  dom.mismatchChart.innerHTML = [
    mismatchRowMarkup(summary.low_star, "reviews with explicit risk"),
    mismatchRowMarkup(summary.high_star, "reviews with hidden risk"),
  ].join("");

  if (summary.high_star.total > 0) {
    const share = ((summary.hiddenRisk / summary.high_star.total) * 100).toFixed(1);
    dom.mismatchInsight.textContent =
      `${summary.hiddenRisk} of ${summary.high_star.total} high-star reviews still contain negative sentiment `
      + `or issue flags. That is ${share}% hidden risk that star ratings alone would miss.`;
    return;
  }

  dom.mismatchInsight.textContent =
    "No 4-5 star reviews are present in the current batch, so hidden-risk detection is not applicable yet.";
}

function renderKpis(summary) {
  if (!dom.kpiGrid) return;
  const items = [
    ["&#128229;", "Inputs", summary.total ?? 0, ""],
    ["&#128681;", "Flagged", summary.flagged ?? 0, "flagged"],
    ["&#9888;&#65039;", "Negative", summary.negative ?? 0, "negative"],
    ["&#128993;", "Uncertain", summary.uncertain ?? 0, "uncertain"],
    ["&#9989;", "Positive", summary.positive ?? 0, "good"],
  ];

  dom.kpiGrid.innerHTML = items
    .map(
      ([icon, label, value, tone]) => `
      <article class="kpi-card ${tone}">
        <div class="kpi-card-head">
          <span class="kpi-card-icon" aria-hidden="true">${icon}</span>
          <h4>${label}</h4>
        </div>
        <p>${value}</p>
        <span>current batch</span>
      </article>
    `
    )
    .join("");
}

function renderCustomerSay(summary, issueRows) {
  if (!dom.customerSayText || !dom.customerSayTags) return;
  const total = summary?.total ?? 0;
  const flagged = summary?.flagged ?? 0;
  const uncertain = summary?.uncertain ?? 0;
  const automationRate = total > 0
    ? (((total - uncertain) / total) * 100).toFixed(1)
    : "0.0";

  let sentence = "Owner insight will be generated from the current batch once data loading is complete.";
  if (total > 0) {
    sentence =
      `Automation coverage is ${automationRate}% for the active batch. `
      + `${flagged} reviews are flagged and ${uncertain} remain uncertain for manual follow-up.`;
  }
  dom.customerSayText.textContent = sentence;

  const tags = issueRows
    .filter((row) => row.count > 0)
    .slice(0, 6)
    .map((row) => `${row.label} (${row.count})`);

  if (!tags.length) tags.push("no issue spikes", "stable feedback");
  dom.customerSayTags.innerHTML = tags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("");
}

function renderDefaultSelection(rows) {
  if (currentView !== "admin") {
    renderSingleReviewAnalysis(null);
    return;
  }

  const queueRows = buildAttentionQueueRows(Array.isArray(rows) ? rows : []);
  if (!queueRows.length) {
    renderSingleReviewAnalysis(null);
    return;
  }

  renderSingleReviewAnalysis(queueRows[0], 1);
}

function updateReviewsSectionChrome() {
  if (dom.reviewsSectionTitle) {
    dom.reviewsSectionTitle.textContent = currentView === "admin"
      ? "Attention Queue"
      : "Recent Customer Reviews";
  }

  if (dom.reviewsSectionNote) {
    dom.reviewsSectionNote.textContent = currentView === "admin"
      ? "Prioritized by risk score for CSKH triage and ticket routing."
      : "Customer-facing review feed.";
  }
}

function setView(view) {
  currentView = view === "admin" ? "admin" : "customer";
  document.body.setAttribute("data-app-view", currentView);

  dom.customerViewBtn?.classList.toggle("active", currentView === "customer");
  dom.adminViewBtn?.classList.toggle("active", currentView === "admin");

  if (dom.viewSwitchText) {
    dom.viewSwitchText.textContent = VIEW_COPY[currentView];
  }

  updateReviewsSectionChrome();
  renderBatchReviewFeed(currentPredictions);
  renderDefaultSelection(currentPredictions);
}

function showAnalyticsPanels(show) {
  if (!dom.summaryPanel || !dom.distributionPanel || !dom.metricsPanel || !dom.triageFocusPanel) return;
  const method = show ? "remove" : "add";
  dom.summaryPanel.classList[method]("hidden");
  dom.distributionPanel.classList[method]("hidden");
  dom.metricsPanel.classList[method]("hidden");
  dom.triageFocusPanel.classList[method]("hidden");
}

function prependManualReview(text, rating) {
  const row = {
    id: `manual_${Date.now()}`,
    rating: normalizedRating(rating),
    text: String(text || "").trim(),
    issue_flags: emptyIssueFlags(),
  };

  if (!row.text) return false;
  datasetReviews = [row, ...datasetReviews];
  currentBatchRows = [row, ...currentBatchRows].slice(0, BATCH_SIZE);
  return true;
}

function nonEmptyTextsFromRows(rows) {
  return rows.map((row) => String(row.text || "").trim()).filter(Boolean);
}

async function fetchStatus() {
  try {
    const response = await fetch("/api/status?include_transformer=false");
    if (!response.ok) throw new Error(`Status endpoint failed (${response.status})`);
    currentStatus = await response.json();
    renderStatus(currentStatus);
  } catch (error) {
    setMessage(`Cannot load model status: ${error.message}`, true);
  }
}

async function fetchCatalog() {
  try {
    const response = await fetch("/api/catalog");
    if (!response.ok) throw new Error(`Catalog endpoint failed (${response.status})`);
    const data = await response.json();
    catalogItems = Array.isArray(data.items) ? data.items : [];
  } catch (error) {
    catalogItems = [];
    setMessage(`Cannot load catalog images: ${error.message}`, true);
  }

  renderProductGallery(catalogItems);
  renderCatalogContext(catalogItems);
  renderRecommendedProducts(catalogItems);
}

async function requestPredictChunk(chunkRows) {
  const texts = nonEmptyTextsFromRows(chunkRows);
  if (!texts.length) return { predictions: [], status: null };

  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts, include_transformer: false }),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `Prediction request failed (${response.status}).`);
  }

  const predictions = Array.isArray(data.predictions) ? data.predictions : [];
  if (predictions.length !== texts.length) throw new Error("Prediction response size mismatch.");

  return { predictions, status: data.status || null };
}

async function predictRowsRobust(rows) {
  const mergedPredictions = [];
  const errors = [];
  let apiStatus = null;
  let fallbackChunks = 0;
  let chunkCount = 0;

  for (let i = 0; i < rows.length; i += PREDICT_CHUNK_SIZE) {
    const chunkRows = rows.slice(i, i + PREDICT_CHUNK_SIZE);
    if (!chunkRows.length) continue;

    chunkCount += 1;
    try {
      const chunkResult = await requestPredictChunk(chunkRows);
      mergedPredictions.push(...chunkResult.predictions);
      if (!apiStatus && chunkResult.status) apiStatus = chunkResult.status;
    } catch (error) {
      fallbackChunks += 1;
      errors.push(error.message || String(error));
      mergedPredictions.push(...buildFallbackPredictions(chunkRows));
    }
  }

  return {
    predictions: mergedPredictions,
    status: apiStatus,
    fallbackChunks,
    chunkCount,
    errors,
  };
}

async function analyzeCurrentBatch() {
  const texts = nonEmptyTextsFromRows(currentBatchRows);
  if (!texts.length) {
    currentPredictions = [];
    renderBatchReviewFeed([]);
    renderSingleReviewAnalysis(null);
    renderOpsPieChart([]);
    renderRailSnapshot({}, [], []);
    renderTriageFocus({}, [], []);
    renderMismatchChart([]);
    showAnalyticsPanels(false);
    setMessage("No reviews available in current batch.", true);
    return;
  }

  try {
    setMessage(`Analyzing ${texts.length} reviews from current batch...`);
    const result = await predictRowsRobust(currentBatchRows);

    currentPredictions = result.predictions.length
      ? attachSourceMetadata(currentBatchRows, result.predictions)
      : buildFallbackPredictions(currentBatchRows);

    if (result.status) {
      currentStatus = result.status;
      renderStatus(currentStatus);
    }

    const summary = buildSummaryFromPredictions(currentPredictions);
    const issueRows = buildIssueRowsFromPredictions(currentPredictions);

    renderOpsPieChart(issueRows);
    renderRailSnapshot(summary, issueRows, currentPredictions);
    renderTriageFocus(summary, issueRows, currentPredictions);
    renderMismatchChart(currentPredictions);
    renderKpis(summary);
    renderCustomerSay(summary, issueRows);
    renderBatchReviewFeed(currentPredictions);
    renderDefaultSelection(currentPredictions);
    showAnalyticsPanels(true);

    if (result.fallbackChunks === 0) {
      setMessage(`Done. Current batch analyzed: ${texts.length} reviews.`);
    } else {
      setMessage(
        `Done with fallback on ${result.fallbackChunks}/${result.chunkCount} chunks.`,
        true
      );
    }
  } catch (error) {
    currentPredictions = buildFallbackPredictions(currentBatchRows);
    const summary = buildSummaryFromPredictions(currentPredictions);
    const issueRows = buildIssueRowsFromPredictions(currentPredictions);

    renderOpsPieChart(issueRows);
    renderRailSnapshot(summary, issueRows, currentPredictions);
    renderTriageFocus(summary, issueRows, currentPredictions);
    renderMismatchChart(currentPredictions);
    renderKpis(summary);
    renderCustomerSay(summary, issueRows);
    renderBatchReviewFeed(currentPredictions);
    renderDefaultSelection(currentPredictions);
    showAnalyticsPanels(true);

    setMessage(`Auto-analysis fallback: ${error.message}`, true);
  }
}

async function fetchReviewPool() {
  try {
    const response = await fetch("/api/review_pool?limit=1000");
    if (!response.ok) throw new Error(`Review pool endpoint failed (${response.status})`);

    const data = await response.json();
    datasetReviews = Array.isArray(data.reviews) ? data.reviews : [];
    currentBatchRows = datasetReviews.slice(0, BATCH_SIZE);
  } catch (error) {
    datasetReviews = [];
    currentBatchRows = [];
    setMessage(`Cannot load review dataset: ${error.message}`, true);
  }

  refreshRatingPanels();
  await analyzeCurrentBatch();
}

function attachEvents() {
  dom.customerViewBtn?.addEventListener("click", () => {
    setView("customer");
  });

  dom.adminViewBtn?.addEventListener("click", () => {
    setView("admin");
  });

  dom.dateRangeSelect?.addEventListener("change", async () => {
    currentDateRange = String(dom.dateRangeSelect?.value || "today");
    renderDateContext();
    setMessage(`Date context switched to ${dateRangeLabel()}. Reloading current batch...`);
    await fetchReviewPool();
    setMessage(
      `Date context now set to ${dateRangeLabel()}. Current dataset has no timestamp column, so this remains a truthful mock reload.`
    );
  });

  dom.runSampleFromLeftBtn?.addEventListener("click", async () => {
    setMessage("Refreshing current batch...");
    await fetchReviewPool();
  });

  dom.writeReviewBtn?.addEventListener("click", () => {
    const panelIsOpen = dom.reviewComposePanel && !dom.reviewComposePanel.classList.contains("hidden");
    if (panelIsOpen) {
      closeComposePanel(false);
      return;
    }
    openComposePanel();
  });

  dom.reviewComposeCancel?.addEventListener("click", () => {
    closeComposePanel(true);
  });

  dom.reviewComposeSubmit?.addEventListener("click", async () => {
    const text = String(dom.reviewComposeText?.value || "").trim();
    const rating = normalizedRating(dom.reviewComposeRating?.value || 5);

    if (text.length < 6) {
      setComposeError("Please write at least 6 characters.");
      return;
    }

    if (!prependManualReview(text, rating)) {
      setComposeError("Could not add this review. Please try again.");
      return;
    }

    closeComposePanel(true);
    refreshRatingPanels();
    await analyzeCurrentBatch();
    dom.reviewsFeed?.scrollIntoView({ behavior: "smooth", block: "start" });
    setMessage("Review added and analyzed in current batch.");
  });

  dom.reviewComposeText?.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeComposePanel(false);
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      dom.reviewComposeSubmit?.click();
    }
  });

  dom.promptChips?.forEach((button) => {
    button.addEventListener("click", () => {
      openComposePanel();
      if (dom.reviewComposeText) {
        dom.reviewComposeText.value = String(button.getAttribute("data-prompt-text") || "");
      }
      if (dom.reviewComposeRating) {
        dom.reviewComposeRating.value = String(button.getAttribute("data-prompt-rating") || "3");
      }
    });
  });
}

function init() {
  normalizeAdminLayout();
  renderOpsPieChart([]);
  renderRailSnapshot({}, [], []);
  renderTriageFocus({}, [], []);
  renderMismatchChart([]);
  renderBatchReviewFeed([]);
  renderSingleReviewAnalysis(null);
  renderCustomerSay({}, []);
  relocateAdminToolbar();
  refreshRatingPanels();
  renderDateContext();
  updateReviewsSectionChrome();
  syncComposeButton();

  attachEvents();
  setView(currentView);
  fetchStatus();
  fetchCatalog();
  fetchReviewPool();
}

init();

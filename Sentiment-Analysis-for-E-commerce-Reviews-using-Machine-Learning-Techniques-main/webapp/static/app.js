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

const dom = {
  adminViewBtn: document.getElementById("admin-view-btn"),
  customerViewBtn: document.getElementById("customer-view-btn"),
  viewSwitchText: document.getElementById("view-switch-text"),
  statusStrip: document.getElementById("status-strip"),
  mainProductImage: document.getElementById("main-product-image"),
  productThumbGrid: document.getElementById("product-thumb-grid"),
  recommendedGrid: document.getElementById("recommended-grid"),
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
  runMessage: document.getElementById("run-message"),
  summaryPanel: document.getElementById("summary-panel"),
  distributionPanel: document.getElementById("distribution-panel"),
  issuesPanel: document.getElementById("issues-panel"),
  kpiGrid: document.getElementById("kpi-grid"),
  issuesTableBody: document.querySelector("#issues-table tbody"),
  reviewsSectionTitle: document.getElementById("reviews-section-title"),
  reviewsSectionNote: document.getElementById("reviews-section-note"),
  opsLabelPie: document.getElementById("ops-label-pie"),
  opsLabelLegend: document.getElementById("ops-label-legend"),
  opsTotalFlags: document.getElementById("ops-total-flags"),
  opsBatchNote: document.getElementById("ops-batch-note"),
  singleReviewHint: document.getElementById("single-review-hint"),
  singleReviewContent: document.getElementById("single-review-content"),
};

let currentView = document.body.getAttribute("data-app-view") || "customer";
let catalogItems = [];
let datasetReviews = [];
let currentBatchRows = [];
let currentPredictions = [];
let currentStatus = null;

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

function closeComposePanel(resetFields = false) {
  if (!dom.reviewComposePanel) return;
  dom.reviewComposePanel.classList.add("hidden");
  setComposeError("");
  if (!resetFields) return;
  if (dom.reviewComposeText) dom.reviewComposeText.value = "";
  if (dom.reviewComposeRating) dom.reviewComposeRating.value = "5";
}

function openComposePanel() {
  if (!dom.reviewComposePanel) return;
  dom.reviewComposePanel.classList.remove("hidden");
  setComposeError("");
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
    dom.leftOverallScore.textContent = "4.7 out of 5";
    dom.leftRatingCount.textContent = "0 global ratings";
    return;
  }

  const rounded = Math.max(1, Math.min(5, Math.round(summary.avg)));
  dom.productStars.innerHTML = starText(rounded);
  dom.productRatingValue.textContent = `${summary.avg.toFixed(1)} out of 5`;
  dom.productRatingCount.textContent = `(${summary.count} reviews)`;
  dom.leftOverallScore.textContent = `${summary.avg.toFixed(1)} out of 5`;
  dom.leftRatingCount.textContent = `${summary.count} global ratings`;
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
  const summary = starSummaryFromDataset(datasetReviews);
  renderProductRating(summary);
  renderRatingBars(summary);
}

function renderStatus(status) {
  if (!dom.statusStrip) return;
  const classic = status?.classic || {};
  const transformer = status?.transformer || {};
  const modelInfo = classic.model_info || {};
  const classicLoaded = Boolean(classic.loaded);
  const transformerLoaded = Boolean(transformer.loaded);
  const cards = [
    {
      label: "Status",
      value: classicLoaded ? "PRODUCTION MODE" : "MODEL UNAVAILABLE",
      tone: classicLoaded ? "production" : "",
    },
    {
      label: "Classic Runtime",
      value: classic.message || (classicLoaded ? "ready" : "unavailable"),
      tone: classicLoaded ? "inference" : "",
    },
    {
      label: "Issue Mode",
      value: classic.issue_mode || "N/A",
      tone: classicLoaded ? "inference" : "",
    },
    {
      label: "Transformer",
      value: transformerLoaded ? "ACTIVE" : (transformer.message || "optional"),
      tone: transformerLoaded ? "transformer" : "",
    },
    { label: "Classic K*", value: modelInfo.k_features ?? "N/A" },
    { label: "Thresholds", value: modelInfo.thresholds ?? "N/A" },
    { label: "Variant", value: modelInfo.variant ?? "N/A" },
    { label: "Model Timestamp", value: modelInfo.trained_at ?? "N/A" },
  ];

  dom.statusStrip.innerHTML = cards
    .map(
      ({ label, value, tone }) => `
      <article class="status-card${tone ? ` status-card--${tone}` : ""}">
        <p class="status-label">${label}</p>
        <p class="status-value${tone ? ` status-value--${tone}` : ""}">${value}</p>
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

function renderSingleReviewAnalysis(row, displayIndex = 0) {
  if (!dom.singleReviewContent || !dom.singleReviewHint) return;

  if (!row || currentView !== "admin") {
    dom.singleReviewHint.textContent =
      "Click Analyze on a queue item to inspect prediction details for that review.";
    dom.singleReviewContent.innerHTML = "";
    return;
  }

  dom.singleReviewHint.textContent = `Showing analysis for queue item #${displayIndex}.`;
  const workflowHtml = hasWorkflowEscalation(row)
    ? `
    <div class="workflow-block">
      <span class="workflow-badge">${WORKFLOW_COPY.badge}</span>
      <button type="button" class="prime-btn jira-action-btn">${WORKFLOW_COPY.action}</button>
      <p class="workflow-meta">${WORKFLOW_COPY.context}</p>
    </div>
  `
    : "";
  dom.singleReviewContent.innerHTML = `
    <table class="single-review-table">
      <tbody>
        <tr><th>Label</th><td>${labelPill(row.classic_label)}</td></tr>
        <tr><th>Probability</th><td>${row.classic_probability == null ? "N/A" : Number(row.classic_probability).toFixed(3)}</td></tr>
        <tr><th>Risk Score</th><td>${escapeHtml(String(row.risk_score ?? "N/A"))}</td></tr>
        <tr><th>Issue Labels</th><td>${escapeHtml(row.issue_summary || "-")}</td></tr>
        <tr><th>Review</th><td class="single-review-text-cell">${escapeHtml(row.text || "")}</td></tr>
      </tbody>
    </table>
    ${workflowHtml}
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
  dom.reviewsFeed.innerHTML = displayedRows
    .map((row, idx) => {
      const stars = starCountFromLabel(row.classic_label, row.classic_probability);
      if (currentView !== "admin") {
        return `
        <article class="review-card public-review-card">
          <div class="review-head-main">
            <div class="avatar">${idx + 1}</div>
            <div>
              <p class="review-title">${reviewHeadline(row.classic_label)}</p>
              <p class="review-meta">Verified customer review | ${starText(stars)}</p>
            </div>
          </div>
          <p class="review-text">${escapeHtml(row.text || "")}</p>
        </article>
        `;
      }

      return `
      <article class="review-card queue-card">
        <div class="review-head">
          <div class="review-head-main">
            <div class="avatar">${idx + 1}</div>
            <div>
              <p class="review-title">${reviewHeadline(row.classic_label)}</p>
              <p class="review-meta">Confidence: ${row.classic_confidence || "N/A"} | Risk: ${row.risk_score ?? "N/A"}</p>
            </div>
          </div>
          <button type="button" class="review-analyze-btn" data-review-index="${idx}">Analyze</button>
        </div>
        <p class="review-stars">${labelPill(row.classic_label)}</p>
        <p class="review-text">${escapeHtml(row.text || "")}</p>
        <p class="review-meta">Issue labels: ${escapeHtml(row.issue_summary || "-")}</p>
      </article>
      `;
    })
    .join("");

  if (currentView === "admin") {
    const buttons = dom.reviewsFeed.querySelectorAll(".review-analyze-btn");
    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        const idx = Number(button.getAttribute("data-review-index"));
        renderSingleReviewAnalysis(displayedRows[idx], idx + 1);
      });
    });
  }
}

function buildFallbackPredictions(rows) {
  return rows.map((row) => {
    const rating = normalizedRating(row.rating);
    const label = sentimentFromRating(rating);
    const flags = normalizeIssueFlags(row);
    const issueCount = activeIssueKeys(flags).length;
    return {
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

function renderKpis(summary) {
  if (!dom.kpiGrid) return;
  const items = [
    ["Inputs", summary.total ?? 0, ""],
    ["Flagged", summary.flagged ?? 0, "flagged"],
    ["Negative", summary.negative ?? 0, "negative"],
    ["Uncertain", summary.uncertain ?? 0, "uncertain"],
    ["Positive", summary.positive ?? 0, "good"],
  ];

  dom.kpiGrid.innerHTML = items
    .map(
      ([label, value, tone]) => `
      <article class="kpi-card ${tone}">
        <h4>${label}</h4>
        <p>${value}</p>
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
      ? "Prioritized by risk score for CSKH triage."
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

function renderIssues(issueRows) {
  if (!dom.issuesTableBody) return;

  const rows = issueRows.filter((row) => Number(row.count) > 0);
  if (!rows.length) {
    dom.issuesTableBody.innerHTML = `<tr><td>No issue labels in current batch.</td></tr>`;
    return;
  }

  dom.issuesTableBody.innerHTML = rows
    .map(
      (row) => `
      <tr>
        <td>${escapeHtml(row.label)}</td>
      </tr>
    `
    )
    .join("");
}

function showAnalyticsPanels(show) {
  if (!dom.summaryPanel || !dom.distributionPanel || !dom.issuesPanel) return;
  const method = show ? "remove" : "add";
  dom.summaryPanel.classList[method]("hidden");
  dom.distributionPanel.classList[method]("hidden");
  dom.issuesPanel.classList[method]("hidden");
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
    showAnalyticsPanels(false);
    setMessage("No reviews available in current batch.", true);
    return;
  }

  try {
    setMessage(`Analyzing ${texts.length} reviews from current batch...`);
    const result = await predictRowsRobust(currentBatchRows);

    currentPredictions = result.predictions.length
      ? result.predictions
      : buildFallbackPredictions(currentBatchRows);

    if (result.status) {
      currentStatus = result.status;
      renderStatus(currentStatus);
    }

    const summary = buildSummaryFromPredictions(currentPredictions);
    const issueRows = buildIssueRowsFromPredictions(currentPredictions);

    renderOpsPieChart(issueRows);
    renderKpis(summary);
    renderCustomerSay(summary, issueRows);
    renderBatchReviewFeed(currentPredictions);
    renderIssues(issueRows);
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
    renderKpis(summary);
    renderCustomerSay(summary, issueRows);
    renderBatchReviewFeed(currentPredictions);
    renderIssues(issueRows);
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
}

function init() {
  renderOpsPieChart([]);
  renderBatchReviewFeed([]);
  renderSingleReviewAnalysis(null);
  renderCustomerSay({}, []);
  refreshRatingPanels();
  updateReviewsSectionChrome();

  attachEvents();
  setView(currentView);
  fetchStatus();
  fetchCatalog();
  fetchReviewPool();
}

init();

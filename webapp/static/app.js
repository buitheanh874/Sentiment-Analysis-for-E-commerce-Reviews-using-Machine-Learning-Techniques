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

const dom = {
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
  datasetReviewCount: document.getElementById("dataset-review-count"),
  datasetReviewSource: document.getElementById("dataset-review-source"),
  datasetReviewsFeed: document.getElementById("dataset-reviews-feed"),
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
  mismatchNote: document.getElementById("mismatch-note"),
  opsLabelPie: document.getElementById("ops-label-pie"),
  opsLabelLegend: document.getElementById("ops-label-legend"),
  opsTotalFlags: document.getElementById("ops-total-flags"),
  opsBatchNote: document.getElementById("ops-batch-note"),
  singleReviewHint: document.getElementById("single-review-hint"),
  singleReviewContent: document.getElementById("single-review-content"),
};

let currentPredictions = [];
let catalogItems = [];
let datasetReviews = [];
let datasetSource = "";
let currentBatchRows = [];

function emptyIssueFlags() {
  const flags = {};
  ISSUE_LABELS.forEach((item) => {
    flags[item.key] = 0;
  });
  return flags;
}

function setMessage(text, isError = false) {
  if (!dom.runMessage) return;
  dom.runMessage.textContent = text;
  dom.runMessage.style.color = isError ? "#a4371f" : "";
}

function formatProb(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A";
  }
  return Number(value).toFixed(3);
}

function labelPill(label) {
  const normalized = String(label || "").toLowerCase();
  return `<span class="pill ${normalized}">${label || "N/A"}</span>`;
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
  const currentLabel = String(label || "");
  if (currentLabel === "POSITIVE") return 5;
  if (currentLabel === "NEGATIVE") return 1;
  if (currentLabel === "NEEDS_ATTENTION") return 2;
  if (currentLabel === "UNCERTAIN") return 3;
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

function normalizeIssueFlags(row) {
  const source = row?.issue_flags || {};
  const flags = {};
  ISSUE_LABELS.forEach((item) => {
    const raw = source[item.key] ?? row?.[item.key] ?? 0;
    flags[item.key] = Number(raw) >= 1 ? 1 : 0;
  });
  return flags;
}

function activeIssueKeys(issueFlags) {
  return ISSUE_LABELS.filter((item) => Number(issueFlags[item.key]) >= 1).map((item) => item.key);
}

function issueSummaryFromFlags(issueFlags) {
  const active = activeIssueKeys(issueFlags);
  if (!active.length) return "-";
  return active.join(", ");
}

function issueCountsFromRows(rows) {
  const counts = {};
  ISSUE_LABELS.forEach((item) => {
    counts[item.key] = 0;
  });

  rows.forEach((row) => {
    const flags = normalizeIssueFlags(row);
    ISSUE_LABELS.forEach((item) => {
      counts[item.key] += Number(flags[item.key]) >= 1 ? 1 : 0;
    });
  });
  return counts;
}

function totalIssueHits(counts) {
  return ISSUE_LABELS.reduce((acc, item) => acc + Number(counts[item.key] || 0), 0);
}

function renderOpsPieChart(rows) {
  if (!dom.opsLabelPie || !dom.opsLabelLegend || !dom.opsTotalFlags || !dom.opsBatchNote) return;

  const counts = issueCountsFromRows(rows);
  const total = totalIssueHits(counts);
  dom.opsTotalFlags.textContent = String(total);
  dom.opsBatchNote.textContent = `Current batch: ${rows.length} reviews | Total issue signals: ${total}`;

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
  const modelInfo = status?.classic?.model_info || {};
  const cards = [
    ["Classic Status", status?.classic?.message ?? "ready"],
    ["Classic K*", modelInfo.k_features ?? "N/A"],
    ["Thresholds", modelInfo.thresholds ?? "N/A"],
    ["Variant", modelInfo.variant ?? "N/A"],
    ["Issue Mode", status?.classic?.issue_mode ?? "N/A"],
    ["Transformer", status?.transformer?.message ?? "disabled"],
    ["Model Timestamp", modelInfo.trained_at ?? "N/A"],
  ];
  dom.statusStrip.innerHTML = cards
    .map(
      ([label, value]) => `
      <article class="status-card">
        <p class="status-label">${label}</p>
        <p class="status-value">${value}</p>
      </article>
    `
    )
    .join("");
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
  if (resetFields) {
    if (dom.reviewComposeText) dom.reviewComposeText.value = "";
    if (dom.reviewComposeRating) dom.reviewComposeRating.value = "5";
  }
}

function openComposePanel() {
  if (!dom.reviewComposePanel) return;
  dom.reviewComposePanel.classList.remove("hidden");
  setComposeError("");
  if (dom.reviewComposeText) {
    dom.reviewComposeText.focus();
  }
}

function prependManualReview(text, rating) {
  const row = {
    id: `manual_${Date.now()}`,
    rating: normalizedRating(rating),
    text: String(text || "").trim(),
    issue_flags: emptyIssueFlags(),
  };

  if (!row.text) return false;
  datasetSource = datasetSource || "manual_input";
  datasetReviews = [row, ...datasetReviews];
  currentBatchRows = [row, ...currentBatchRows].slice(0, BATCH_SIZE);
  return true;
}

function niceNameFromFile(fileName, index) {
  const base = String(fileName || "").replace(/\.[^.]+$/, "");
  if (!base) return `Recommended Item ${index + 1}`;
  const short = base.slice(0, 26);
  return `NLP Book Item ${index + 1} - ${short}`;
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
      const price = (349000 + idx * 37000).toLocaleString("vi-VN");
      return `
      <article class="recommended-card">
        <img src="${item.url}" alt="${item.name}" />
        <p class="recommended-title">${niceNameFromFile(item.name, idx)}</p>
        <p class="recommended-meta">Prime eligible | Free returns</p>
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
      thumbs.forEach((x) => x.classList.remove("active"));
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

function sourceFileName(pathValue) {
  const path = String(pathValue || "");
  if (!path) return "";
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderSingleReviewAnalysis(row, displayIndex = 0) {
  if (!dom.singleReviewContent || !dom.singleReviewHint) return;
  if (!row) {
    dom.singleReviewHint.textContent =
      "Click Analyze on a review card to inspect prediction details for that exact review.";
    dom.singleReviewContent.innerHTML = "";
    return;
  }

  dom.singleReviewHint.textContent = `Showing analysis for selected review #${displayIndex}.`;
  dom.singleReviewContent.innerHTML = `
    <table class="single-review-table">
      <tbody>
        <tr><th>Label</th><td>${labelPill(row.classic_label)}</td></tr>
        <tr><th>Confidence</th><td>${escapeHtml(row.classic_confidence || "N/A")} | P(+): ${formatProb(row.classic_probability)}</td></tr>
        <tr><th>Risk Score</th><td>${Number(row.risk_score ?? 0).toFixed(1)}</td></tr>
        <tr><th>Issue Labels</th><td>${escapeHtml(row.issue_summary || "-")}</td></tr>
        <tr><th>Review Text</th><td><div class="single-review-text-cell">${escapeHtml(row.text || "")}</div></td></tr>
      </tbody>
    </table>
  `;
}

function renderBatchReviewFeed(rows) {
  if (!dom.reviewsFeed) return;
  if (!rows.length) {
    dom.reviewsFeed.innerHTML = `<p class="muted">No reviews in current batch.</p>`;
    renderSingleReviewAnalysis(null);
    return;
  }

  const displayedRows = rows.slice(0, 12);
  dom.reviewsFeed.innerHTML = displayedRows
    .map((row, idx) => {
      const stars = starCountFromLabel(row.classic_label, row.classic_probability);
      const reason = row.fallback_reason && row.fallback_reason !== "-" ? row.fallback_reason : "model_default";
      return `
      <article class="review-card">
        <div class="review-head">
          <div class="review-head-main">
            <div class="avatar">${idx + 1}</div>
            <div>
              <p class="review-title">${reviewHeadline(row.classic_label)}</p>
              <p class="review-meta">Current batch inference | Confidence: ${row.classic_confidence || "N/A"}</p>
            </div>
          </div>
          <button type="button" class="review-analyze-btn" data-review-index="${idx}">Analyze</button>
        </div>
        <p class="review-stars">${starText(stars)} <span class="review-meta">${row.classic_label || "N/A"}</span></p>
        <p class="review-text">${escapeHtml(row.text || "")}</p>
        <p class="review-meta">Issue labels: ${escapeHtml(row.issue_summary || "-")} | Reason: ${escapeHtml(reason)}</p>
      </article>
      `;
    })
    .join("");

  const buttons = dom.reviewsFeed.querySelectorAll(".review-analyze-btn");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const idx = Number(button.getAttribute("data-review-index"));
      const selected = displayedRows[idx];
      renderSingleReviewAnalysis(selected, idx + 1);
    });
  });
}

function renderDatasetReviewFeed(rows) {
  if (!dom.datasetReviewsFeed || !dom.datasetReviewCount || !dom.datasetReviewSource) return;
  const shownRows = rows.slice(0, 1000);
  const sourceName = sourceFileName(datasetSource);
  dom.datasetReviewSource.textContent = sourceName
    ? `Source dataset: ${sourceName}`
    : "Source dataset: not found";
  dom.datasetReviewCount.textContent = `${shownRows.length} shown / ${rows.length} loaded`;

  if (!shownRows.length) {
    dom.datasetReviewsFeed.innerHTML = `<p class="muted">No dataset reviews available.</p>`;
    return;
  }

  dom.datasetReviewsFeed.innerHTML = shownRows
    .map((row, idx) => {
      const stars = normalizedRating(row.rating);
      const reviewId = row.id ? String(row.id) : `row_${idx + 1}`;
      return `
      <article class="review-card">
        <div class="review-head">
          <div class="review-head-main">
            <div class="avatar">D${idx + 1}</div>
            <div>
              <p class="review-title">Historical customer review</p>
              <p class="review-meta">CSV id: ${escapeHtml(reviewId)}</p>
            </div>
          </div>
        </div>
        <p class="review-stars">${starText(stars)} <span class="review-meta">${stars} / 5</span></p>
        <p class="review-text">${escapeHtml(row.text || "")}</p>
      </article>
      `;
    })
    .join("");
}

function buildFallbackPredictions(rows) {
  return rows.map((row) => {
    const rating = normalizedRating(row.rating);
    const label = sentimentFromRating(rating);
    const flags = normalizeIssueFlags(row);
    const issueSummary = issueSummaryFromFlags(flags);
    const issueCount = activeIssueKeys(flags).length;
    return {
      text: row.text || "",
      classic_label: label,
      classic_probability: rating / 5,
      classic_confidence: "Dataset",
      fallback_reason: "dataset_seed",
      issue_summary: issueSummary,
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

function issueRowsFromBatch(rows) {
  const counts = issueCountsFromRows(rows);
  const total = totalIssueHits(counts);
  return ISSUE_LABELS.map((item) => {
    const count = Number(counts[item.key] || 0);
    const share = total > 0 ? Number((count / total).toFixed(3)) : 0;
    return {
      label: item.key,
      count,
      share,
    };
  }).sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
}

function mergeDatasetIssueHints(predictions, sourceRows) {
  if (!Array.isArray(predictions) || !Array.isArray(sourceRows) || !predictions.length) {
    return predictions || [];
  }

  return predictions.map((row, idx) => {
    const source = sourceRows[idx];
    if (!source) return row;

    const sourceFlags = normalizeIssueFlags(source);
    const sourceSummary = issueSummaryFromFlags(sourceFlags);
    if (!sourceSummary || sourceSummary === "-") {
      return row;
    }

    const currentSummary = String(row.issue_summary || "-").trim();
    if (currentSummary && currentSummary !== "-") {
      return row;
    }

    const sourceIssueCount = activeIssueKeys(sourceFlags).length;
    const existingReason = String(row.fallback_reason || "-");
    const nextReason = existingReason && existingReason !== "-" ? existingReason : "dataset_issue_hint";

    return {
      ...row,
      issue_summary: sourceSummary,
      issue_count: sourceIssueCount,
      fallback_reason: nextReason,
    };
  });
}

function renderKpis(summary) {
  if (!dom.kpiGrid) return;
  const items = [
    ["Inputs", summary.total ?? 0, ""],
    ["Flagged", summary.flagged ?? 0, "alert"],
    ["Negative", summary.negative ?? 0, "alert"],
    ["Uncertain", summary.uncertain ?? 0, ""],
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
  const flaggedRate = total > 0 ? Math.round((flagged / total) * 100) : 0;
  const positive = summary?.positive ?? 0;

  let sentence =
    "Owner insight will be generated from the current batch once data loading is complete.";
  if (total > 0) {
    if (flaggedRate >= 55) {
      sentence = `Current batch shows elevated risk (${flaggedRate}% flagged). Focus on service quality and shipping/refund workflow.`;
    } else if (positive >= flagged) {
      sentence = `Current batch is mostly positive (${positive}/${total}), but recurring issue labels still need monitoring.`;
    } else {
      sentence = `Customer feedback is mixed in this batch. Investigate the top issue labels before scaling sales.`;
    }
  }
  dom.customerSayText.textContent = sentence;

  const tags = issueRows
    .filter((row) => row.count > 0)
    .slice(0, 6)
    .map((row) => `${row.label} (${row.count})`);
  if (!tags.length) {
    tags.push("no issue spikes", "stable feedback");
  }
  dom.customerSayTags.innerHTML = tags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("");
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

async function fetchStatus() {
  try {
    const response = await fetch("/api/status?include_transformer=false");
    if (!response.ok) throw new Error(`Status endpoint failed (${response.status})`);
    const status = await response.json();
    renderStatus(status);
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

function nonEmptyTextsFromRows(rows) {
  return rows.map((row) => String(row.text || "").trim()).filter(Boolean);
}

async function requestPredictChunk(chunkRows) {
  const texts = nonEmptyTextsFromRows(chunkRows);
  if (!texts.length) {
    return { predictions: [], status: null };
  }

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
  if (predictions.length !== texts.length) {
    throw new Error("Prediction response size mismatch.");
  }

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
      if (!apiStatus && chunkResult.status) {
        apiStatus = chunkResult.status;
      }
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
    showAnalyticsPanels(false);
    return;
  }

  try {
    setMessage(`Analyzing ${texts.length} reviews from current batch...`);
    const result = await predictRowsRobust(currentBatchRows);
    currentPredictions = result.predictions;
    if (!currentPredictions.length) {
      currentPredictions = buildFallbackPredictions(currentBatchRows);
    }
    currentPredictions = mergeDatasetIssueHints(currentPredictions, currentBatchRows);

    const issueRows = issueRowsFromBatch(currentBatchRows);
    const summary = buildSummaryFromPredictions(currentPredictions);

    if (result.status) {
      renderStatus(result.status);
    }
    renderKpis(summary);
    renderCustomerSay(summary, issueRows);
    renderBatchReviewFeed(currentPredictions);
    renderIssues(issueRows);
    renderSingleReviewAnalysis(null);

    if (dom.mismatchNote) {
      if (result.fallbackChunks === 0) {
        dom.mismatchNote.textContent = "Click Analyze on each review card to inspect detail.";
      } else if (result.fallbackChunks >= result.chunkCount) {
        dom.mismatchNote.textContent = "Fallback mode from labeled dataset";
      } else {
        dom.mismatchNote.textContent = `Partial fallback: ${result.fallbackChunks}/${result.chunkCount} chunks`;
      }
    }
    showAnalyticsPanels(true);
    if (result.fallbackChunks === 0) {
      setMessage(`Done. Current batch analyzed: ${texts.length} reviews.`);
    } else {
      const firstError = result.errors[0] || "unknown error";
      setMessage(
        `Done with fallback on ${result.fallbackChunks}/${result.chunkCount} chunks. First error: ${firstError}`,
        true
      );
    }
  } catch (error) {
    currentPredictions = buildFallbackPredictions(currentBatchRows);
    const issueRows = issueRowsFromBatch(currentBatchRows);
    const summary = buildSummaryFromPredictions(currentPredictions);

    renderKpis(summary);
    renderCustomerSay(summary, issueRows);
    renderBatchReviewFeed(currentPredictions);
    renderIssues(issueRows);
    renderSingleReviewAnalysis(null);
    if (dom.mismatchNote) {
      dom.mismatchNote.textContent = "Fallback mode from labeled dataset";
    }
    showAnalyticsPanels(true);
    setMessage(`Auto-analysis fallback: ${error.message}`, true);
  }
}

async function fetchReviewPool() {
  try {
    const response = await fetch("/api/review_pool?limit=1000");
    if (!response.ok) throw new Error(`Review pool endpoint failed (${response.status})`);
    const data = await response.json();
    datasetSource = String(data.source || "");
    datasetReviews = Array.isArray(data.reviews) ? data.reviews : [];
    currentBatchRows = datasetReviews.slice(0, BATCH_SIZE);
  } catch (error) {
    datasetSource = "";
    datasetReviews = [];
    currentBatchRows = [];
    setMessage(`Cannot load review dataset: ${error.message}`, true);
  }

  refreshRatingPanels();
  renderDatasetReviewFeed(datasetReviews);
  renderOpsPieChart(currentBatchRows);
  await analyzeCurrentBatch();
}

function attachEvents() {
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

    const inserted = prependManualReview(text, rating);
    if (!inserted) {
      setComposeError("Could not add this review. Please try again.");
      return;
    }

    closeComposePanel(true);
    refreshRatingPanels();
    renderDatasetReviewFeed(datasetReviews);
    renderOpsPieChart(currentBatchRows);
    await analyzeCurrentBatch();
    dom.reviewsFeed?.scrollIntoView({ behavior: "smooth", block: "start" });
    setMessage("Review added and analyzed in current batch.");
  });

  dom.reviewComposeText?.addEventListener("keydown", async (event) => {
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
  renderDatasetReviewFeed([]);
  renderSingleReviewAnalysis(null);
  renderCustomerSay({}, []);
  refreshRatingPanels();
  attachEvents();
  fetchStatus();
  fetchCatalog();
  fetchReviewPool();
}

init();

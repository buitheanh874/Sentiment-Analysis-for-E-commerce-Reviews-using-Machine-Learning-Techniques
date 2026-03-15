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

let catalogItems = [];
let datasetReviews = [];
let currentBatchRows = [];
let currentPredictions = [];

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
  return ISSUE_LABELS.reduce((total, item) => total + Number(counts[item.key] || 0), 0);
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

function niceNameFromFile(fileName, index) {
  const base = String(fileName || "").replace(/\.[^.]+$/, "");
  if (!base) return `Recommended Item ${index + 1}`;
  const short = base.slice(0, 26);
  return `NLP Book Item ${index + 1} - ${short}`;
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
        <tr><th>Issue Labels</th><td>${escapeHtml(row.issue_summary || "-")}</td></tr>
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
      renderSingleReviewAnalysis(displayedRows[idx], idx + 1);
    });
  });
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
    if (!sourceSummary || sourceSummary === "-") return row;

    const currentSummary = String(row.issue_summary || "-").trim();
    if (currentSummary && currentSummary !== "-") return row;

    return {
      ...row,
      issue_summary: sourceSummary,
      issue_count: activeIssueKeys(sourceFlags).length,
      fallback_reason: row.fallback_reason && row.fallback_reason !== "-" ? row.fallback_reason : "dataset_issue_hint",
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
  const positive = summary?.positive ?? 0;
  const flaggedRate = total > 0 ? Math.round((flagged / total) * 100) : 0;

  let sentence = "Owner insight will be generated from the current batch once data loading is complete.";
  if (total > 0) {
    if (flaggedRate >= 55) {
      sentence = `Current batch shows elevated risk (${flaggedRate}% flagged). Focus on service quality and shipping/refund workflow.`;
    } else if (positive >= flagged) {
      sentence = `Current batch is mostly positive (${positive}/${total}), but recurring issue labels still need monitoring.`;
    } else {
      sentence = "Customer feedback is mixed in this batch. Investigate the top issue labels before scaling sales.";
    }
  }
  dom.customerSayText.textContent = sentence;

  const tags = issueRows
    .filter((row) => row.count > 0)
    .slice(0, 6)
    .map((row) => `${row.label} (${row.count})`);

  if (!tags.length) tags.push("no issue spikes", "stable feedback");
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
    renderStatus(await response.json());
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
    currentPredictions = mergeDatasetIssueHints(currentPredictions, currentBatchRows);

    const summary = buildSummaryFromPredictions(currentPredictions);
    const issueRows = issueRowsFromBatch(currentBatchRows);

    if (result.status) renderStatus(result.status);
    renderKpis(summary);
    renderCustomerSay(summary, issueRows);
    renderBatchReviewFeed(currentPredictions);
    renderIssues(issueRows);
    renderSingleReviewAnalysis(null);
    showAnalyticsPanels(true);

    if (dom.mismatchNote) {
      if (result.fallbackChunks === 0) {
        dom.mismatchNote.textContent = "Click Analyze on each review card to inspect detail.";
      } else if (result.fallbackChunks >= result.chunkCount) {
        dom.mismatchNote.textContent = "Fallback mode from labeled dataset";
      } else {
        dom.mismatchNote.textContent = `Partial fallback: ${result.fallbackChunks}/${result.chunkCount} chunks`;
      }
    }

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
    const issueRows = issueRowsFromBatch(currentBatchRows);

    renderKpis(summary);
    renderCustomerSay(summary, issueRows);
    renderBatchReviewFeed(currentPredictions);
    renderIssues(issueRows);
    renderSingleReviewAnalysis(null);
    showAnalyticsPanels(true);
    if (dom.mismatchNote) dom.mismatchNote.textContent = "Fallback mode from labeled dataset";

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

    if (!prependManualReview(text, rating)) {
      setComposeError("Could not add this review. Please try again.");
      return;
    }

    closeComposePanel(true);
    refreshRatingPanels();
    renderOpsPieChart(currentBatchRows);
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

  attachEvents();
  fetchStatus();
  fetchCatalog();
  fetchReviewPool();
}

init();

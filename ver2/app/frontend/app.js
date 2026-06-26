const form = document.querySelector("#predictForm");
const submitBtn = document.querySelector("#submitBtn");
const healthStatus = document.querySelector("#healthStatus");
const emptyState = document.querySelector("#emptyState");
const resultState = document.querySelector("#resultState");
const errorState = document.querySelector("#errorState");
const urlField = document.querySelector("#url");
const titleField = document.querySelector("#title");
const bodyField = document.querySelector("#body");
const upvotesField = document.querySelector("#upvotes");
const commentsField = document.querySelector("#numComments");
const createdUtcField = document.querySelector("#createdUtc");
const translateField = document.querySelector("#translate");
const finalScore = document.querySelector("#finalScore");
const stage1Score = document.querySelector("#stage1Score");
const stage2Score = document.querySelector("#stage2Score");
const labelBadge = document.querySelector("#labelBadge");
const meterFill = document.querySelector("#meterFill");
const cleanTitle = document.querySelector("#cleanTitle");
const cleanBody = document.querySelector("#cleanBody");
const crawlMeta = document.querySelector("#crawlMeta");
const note = document.querySelector("#note");

let mode = "url";

document.querySelectorAll(".mode").forEach((button) => {
  button.addEventListener("click", () => {
    mode = button.dataset.mode;
    document.querySelectorAll(".mode").forEach((b) => b.classList.toggle("active", b === button));
    document.querySelector(".url-only").classList.toggle("hidden", mode !== "url");
    document.querySelector(".manual-only").classList.toggle("hidden", mode !== "manual");
    errorState.classList.add("hidden");
  });
});

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    healthStatus.textContent = data.model_loaded ? "Models loaded" : "Ready";
  } catch {
    healthStatus.textContent = "Backend offline";
  }
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.textContent = isLoading ? "Running inference..." : "Predict";
  healthStatus.textContent = isLoading ? "Scoring" : healthStatus.textContent;
}

function showError(message) {
  errorState.textContent = message;
  errorState.classList.remove("hidden");
  resultState.classList.add("hidden");
  emptyState.classList.add("hidden");
}

function showResult(data) {
  const pFinal = Number(data.p_final_depression_risk);
  const pText = Number(data.p_text_stage1);
  finalScore.textContent = `${(pFinal * 100).toFixed(2)}%`;
  stage1Score.textContent = pText.toFixed(4);
  stage2Score.textContent = pFinal.toFixed(4);
  labelBadge.textContent = `label ${data.predicted_label_at_0_5}`;
  labelBadge.classList.toggle("positive", data.predicted_label_at_0_5 === 1);
  meterFill.style.width = `${Math.max(0, Math.min(100, pFinal * 100))}%`;
  cleanTitle.textContent = data.title_en_clean || "Untitled";
  cleanBody.textContent = data.body_en_clean || "";
  const when = new Date(Number(data.created_utc) * 1000).toLocaleString();
  crawlMeta.textContent = `reactions: ${data.upvotes} · comments: ${data.num_comments} · posted: ${when}`;
  note.textContent = data.note || "";

  emptyState.classList.add("hidden");
  errorState.classList.add("hidden");
  resultState.classList.remove("hidden");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    url: mode === "url" ? urlField.value.trim() || null : null,
    title: mode === "manual" ? titleField.value.trim() : "",
    body: mode === "manual" ? bodyField.value.trim() : "",
    upvotes: Number(upvotesField.value || 0),
    num_comments: Number(commentsField.value || 0),
    created_utc: createdUtcField.value ? Number(createdUtcField.value) : null,
    translate: translateField.checked,
  };

  setLoading(true);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed");
    }
    showResult(data);
    await checkHealth();
  } catch (error) {
    // Fallback: if scraping a URL failed, drop into Manual mode to paste text.
    if (mode === "url") {
      document.querySelector('.mode[data-mode="manual"]').click();
    }
    showError(error.message || "Prediction failed");
  } finally {
    setLoading(false);
  }
});

checkHealth();

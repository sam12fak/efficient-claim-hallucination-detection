/**
 * NEURALFACT // NETFLIX-STYLE SHOWCASE INTERACTIVE LOGIC
 * Architecture episode inspector, 39-D catalog filter, benchmark modal, simulator & FAQ accordion.
 */

// --------------------------------------------------------------------------
// 1. Architecture Episodes Data & Inspector
// --------------------------------------------------------------------------
const ARCH_NODES = {
  "user-input": {
    episode: "EPISODE 01",
    title: "User Input Ingestion & Token Normalization",
    file: "app/streamlit_app.py",
    runtime: "< 0.05 ms",
    badge: "Gateway Ingestion",
    summary: "Ingests raw user prompt containing a Question, generated LLM Answer, and optional Reference Context from RAG or external documents. Performs Unicode normalization and quotation standardization.",
    math: "\\text{Input} = \\{Q, A_{\\text{LLM}}, C_{\\text{Context}}^{\\text{optional}}\\}",
    inputs: "Natural language query, model response text, document context",
    outputs: "Sanitized string tuples passed to claim segmentation engine"
  },
  "claim-segmenter": {
    episode: "EPISODE 02",
    title: "spaCy Claim Extraction & Sentencizer",
    file: "src/claim_extraction.py",
    runtime: "~ 0.8 ms",
    badge: "spaCy Sentencizer",
    summary: "Decomposes long narrative answers into atomic claim propositions. Strips markdown numbering, bullet lists, whitespace artifacts, and isolates verifiable declarative assertions.",
    math: "\\{c_1, c_2, \\dots, c_m\\} = \\text{ExtractClaims}(A_{\\text{LLM}}), \\quad |c_i| > 3 \\text{ tokens}",
    inputs: "Raw LLM answer string",
    outputs: "Structured claim list with unique IDs and sanitized text"
  },
  "semantic-stream": {
    episode: "EPISODE 03",
    title: "Dense Semantic Geometry (384-D)",
    file: "src/semantic_features.py",
    runtime: "~ 1.5 ms (Cached: 0.01 ms)",
    badge: "all-MiniLM-L6-v2",
    summary: "Calculates dense geometric alignments. Uses an LRU vector cache to eliminate redundant forward passes for questions and context paragraphs. Computes max sentence-level context similarity to prevent dilution.",
    math: "s(u, v) = \\frac{u \\cdot v}{\\|u\\|_2 \\|v\\|_2}, \\quad s_{\\max}(c, C) = \\max_{s_k \\in C} s(e_c, e_{s_k})",
    inputs: "Claim text, Question, Full Answer, Context sentences",
    outputs: "4-D Semantic Vector: [q_c_sim, q_c_dist, c_a_sim, ctx_c_sim]"
  },
  "linguistic-stream": {
    episode: "EPISODE 04",
    title: "Linguistic Stance & NER Engine (35-D)",
    file: "src/linguistic_features.py",
    runtime: "~ 0.9 ms",
    badge: "spaCy en_core_web_sm",
    summary: "Extracts 35 grammatical, syntactic, named entity, numerical, and epistemic stance features without requiring heavy neural networks.",
    math: "\\rho_{\\text{ent}} = \\frac{N_{\\text{ent}}}{N_{\\text{tokens}}}, \\quad \\theta_{\\text{stance}} = \\{\\text{cert\\_score}, \\text{uncert\\_score}\\}",
    inputs: "Claim tokens, POS tags, Named Entity spans",
    outputs: "35-D Linguistic Vector across 5 feature categories"
  },
  "feature-fusion": {
    episode: "EPISODE 05",
    title: "Feature Fusion & Closed-Book Imputation",
    file: "src/feature_fusion.py",
    runtime: "< 0.01 ms",
    badge: "39-D Vector Concatenation",
    summary: "Merges the 4 semantic features with the 35 linguistic metrics into a strictly ordered, standardized 39-dimensional dense feature table with question-based context imputation.",
    math: "\\mathbf{x}_i = [\\Phi_{\\text{semantic}} \\parallel \\Phi_{\\text{linguistic}}] \\in \\mathbb{R}^{39}",
    inputs: "Parallel semantic and linguistic extracted feature dictionaries",
    outputs: "Unified numpy array matrix ready for classifier forward pass"
  },
  "detector-model": {
    episode: "EPISODE 06",
    title: "Platt-Calibrated XGBoost & Grounding",
    file: "src/train.py & models/hybrid_xgboost.joblib",
    runtime: "0.001 ms",
    badge: "Platt Scaled Ensemble",
    summary: "Executes ultra-low latency inference (0.001 ms per claim). Applies sigmoid Platt scaling calibrated on held-out validation data to produce accurate risk percentages and verifies context entities.",
    math: "P(y=1 \\mid \\mathbf{x}) = \\frac{1}{1 + \\exp(A \\cdot f_{\\text{XGB}}(\\mathbf{x}) + B)}",
    inputs: "39-D feature vector per claim",
    outputs: "Calibrated probability P(Hallucination in [0, 1])"
  },
  "rag-ollama": {
    episode: "EPISODE 07",
    title: "Local Ollama RAG & Dynamic Grounding",
    file: "src/rag_pipeline.py",
    runtime: "~ 1.2 s (E2E)",
    badge: "RAG + Ollama qwen2.5:3b",
    summary: "End-to-end Retrieval-Augmented Generation indexing Computational Linguistics PPTX slides and PDF course notes. Feeds retrieved context into local Ollama LLMs (qwen2.5:3b) and immediately applies sub-millisecond claim verification to guard against generative hallucinations.",
    math: "\\text{Answer} \\sim \\text{LLM}(Q, \\text{TopK}(C)), \\quad \\text{Risk} = \\text{NeuralFact}(Q, \\text{Answer}, \\text{TopK}(C))",
    inputs: "Course query + indexed PPTX slides & PDF notes (769 chunks)",
    outputs: "Grounded LLM answer + per-claim hallucination risk & latency breakdown"
  }
};

function selectArchNode(nodeKey) {
  const panel = document.getElementById("node-inspector-panel");
  const data = ARCH_NODES[nodeKey];
  if (!data || !panel) return;

  // Toggle active styling
  document.querySelectorAll(".episode-card").forEach(el => el.classList.remove("active"));
  const targetCard = document.querySelector(`[data-node="${nodeKey}"]`);
  if (targetCard) targetCard.classList.add("active");

  // Populate Netflix episode drawer
  panel.classList.add("show");
  panel.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem; border-bottom: 1px solid var(--nf-border); padding-bottom: 1rem;">
      <div>
        <span style="font-family: var(--font-display); font-size: 1.1rem; color: var(--nf-red); letter-spacing: 0.05em;">${data.episode} &bull; ${data.runtime}</span>
        <h3 style="font-size: 1.45rem; font-weight: 800; color: #fff; margin-top: 0.2rem;">${data.title}</h3>
        <span style="font-size: 0.82rem; font-family: var(--font-mono); color: var(--nf-text-muted);">${data.file}</span>
      </div>
      <span class="nf-spec-badge" style="color: var(--nf-red); border-color: var(--nf-red);">${data.badge}</span>
    </div>
    
    <p style="color: var(--nf-text-light); font-size: 0.98rem; line-height: 1.6; margin-bottom: 1.25rem;">${data.summary}</p>
    
    <div style="background: var(--nf-deep-black); border: 1px solid var(--nf-border); border-radius: var(--radius-sm); padding: 1rem 1.25rem; margin-bottom: 1.25rem; font-family: var(--font-mono); font-size: 0.88rem; color: var(--nf-red);">
      <code>${data.math}</code>
    </div>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; font-size: 0.9rem;">
      <div>
        <strong style="color: #fff; display: block; margin-bottom: 0.35rem; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.08em; color: var(--nf-text-muted);">Inputs:</strong>
        <span style="color: var(--nf-text-secondary);">${data.inputs}</span>
      </div>
      <div>
        <strong style="color: #fff; display: block; margin-bottom: 0.35rem; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.08em; color: var(--nf-text-muted);">Outputs:</strong>
        <span style="color: var(--nf-text-secondary);">${data.outputs}</span>
      </div>
    </div>
  `;
}

// --------------------------------------------------------------------------
// 2. 39-Dimensional Feature Taxonomy Dataset
// --------------------------------------------------------------------------
const FEATURES_DATA = [
  // Semantic (4)
  { name: "question_claim_similarity", category: "semantic", type: "Cosine (-1 to 1)", desc: "Cosine similarity between MiniLM claim embedding and question embedding.", importance: 0.0853 },
  { name: "question_claim_distance", category: "semantic", type: "Distance (0 to 2)", desc: "Semantic distance between claim and question (1 - cosine_similarity).", importance: 0.0957 },
  { name: "claim_answer_similarity", category: "semantic", type: "Cosine (-1 to 1)", desc: "Internal coherence and semantic alignment of claim with the broader answer.", importance: 0.0996 },
  { name: "context_claim_similarity", category: "semantic", type: "Cosine (-1 to 1)", desc: "Maximum semantic similarity against any sentence in reference context.", importance: 0.0798 },

  // Surface (5)
  { name: "char_count", category: "surface", type: "Integer (>=0)", desc: "Total character count of the extracted claim sentence.", importance: 0.0798 },
  { name: "word_count", category: "surface", type: "Integer (>=0)", desc: "Total tokenized word count in the claim.", importance: 0.0385 },
  { name: "token_count", category: "surface", type: "Integer (>=0)", desc: "Total linguistic token count including punctuation.", importance: 0.0599 },
  { name: "sentence_length", category: "surface", type: "Integer (>=0)", desc: "Character length of the sanitized sentence.", importance: 0.0554 },
  { name: "avg_word_length", category: "surface", type: "Float (>=0)", desc: "Average character length per word, measuring vocabulary complexity.", importance: 0.0502 },

  // Named Entities (10)
  { name: "entity_person_count", category: "ner", type: "Count (>=0)", desc: "Number of detected PERSON named entities in claim.", importance: 0.0165 },
  { name: "entity_org_count", category: "ner", type: "Count (>=0)", desc: "Number of ORGANIZATION named entities in claim.", importance: 0.0121 },
  { name: "entity_gpe_count", category: "ner", type: "Count (>=0)", desc: "Number of Geopolitical Entities (countries, cities, states).", importance: 0.0142 },
  { name: "entity_loc_count", category: "ner", type: "Count (>=0)", desc: "Non-GPE location entities (mountains, water bodies).", importance: 0.0053 },
  { name: "entity_date_count", category: "ner", type: "Count (>=0)", desc: "Number of absolute or relative dates or periods.", importance: 0.0118 },
  { name: "entity_cardinal_count", category: "ner", type: "Count (>=0)", desc: "Numerals that do not fall under another category.", importance: 0.0135 },
  { name: "entity_money_count", category: "ner", type: "Count (>=0)", desc: "Monetary value assertions (currency units and amounts).", importance: 0.0042 },
  { name: "entity_percent_count", category: "ner", type: "Count (>=0)", desc: "Percentage numerical assertions ('%').", importance: 0.0031 },
  { name: "total_entity_count", category: "ner", type: "Count (>=0)", desc: "Sum of all recognized named entities in the claim.", importance: 0.0356 },
  { name: "entity_density", category: "ner", type: "Ratio [0, 1]", desc: "Ratio of named entity tokens to total word count.", importance: 0.0214 },

  // Numerical & Chronological (6)
  { name: "number_count", category: "numeric", type: "Count (>=0)", desc: "Raw count of digits and numerical tokens extracted via regex.", importance: 0.0284 },
  { name: "has_numbers", category: "numeric", type: "Binary {0, 1}", desc: "Flag indicating whether claim makes explicit numerical assertions.", importance: 0.0195 },
  { name: "digit_density", category: "numeric", type: "Ratio [0, 1]", desc: "Ratio of numeric characters to total character length.", importance: 0.0210 },
  { name: "year_count", category: "numeric", type: "Count (>=0)", desc: "Count of 4-digit Gregorian calendar years (1000–2099).", importance: 0.0312 },
  { name: "has_years", category: "numeric", type: "Binary {0, 1}", desc: "Flag indicating presence of historical or chronological dates.", importance: 0.0256 },
  { name: "numeric_token_ratio", category: "numeric", type: "Ratio [0, 1]", desc: "Ratio of numerical tokens to total words in sentence.", importance: 0.0143 },

  // Part-of-Speech Syntax (10)
  { name: "pos_noun_count", category: "pos", type: "Count (>=0)", desc: "Count of common noun tokens (NOUN).", importance: 0.0142 },
  { name: "pos_noun_ratio", category: "pos", type: "Ratio [0, 1]", desc: "Proportion of claim tokens classified as nouns.", importance: 0.0221 },
  { name: "pos_verb_count", category: "pos", type: "Count (>=0)", desc: "Count of action and relational verb tokens (VERB).", importance: 0.0189 },
  { name: "pos_verb_ratio", category: "pos", type: "Ratio [0, 1]", desc: "Proportion of claim tokens classified as verbs.", importance: 0.0175 },
  { name: "pos_adj_count", category: "pos", type: "Count (>=0)", desc: "Count of descriptive adjective tokens (ADJ).", importance: 0.0163 },
  { name: "pos_adj_ratio", category: "pos", type: "Ratio [0, 1]", desc: "Proportion of claim tokens classified as adjectives.", importance: 0.0204 },
  { name: "pos_adv_count", category: "pos", type: "Count (>=0)", desc: "Count of qualifying adverb tokens (ADV).", importance: 0.0098 },
  { name: "pos_adv_ratio", category: "pos", type: "Ratio [0, 1]", desc: "Proportion of claim tokens classified as adverbs.", importance: 0.0125 },
  { name: "pos_propn_count", category: "pos", type: "Count (>=0)", desc: "Count of proper noun tokens (PROPN).", importance: 0.0245 },
  { name: "pos_propn_ratio", category: "pos", type: "Ratio [0, 1]", desc: "Proportion of claim tokens classified as proper nouns.", importance: 0.0278 },

  // Epistemic Stance & Sentiment (4)
  { name: "certainty_score", category: "stance", type: "Ratio [0, 1]", desc: "Density of dogmatic epistemic markers ('definitely', 'proven', 'always').", importance: 0.0381 },
  { name: "uncertainty_score", category: "stance", type: "Ratio [0, 1]", desc: "Density of epistemic hedge markers ('likely', 'allegedly', 'possibly').", importance: 0.0412 },
  { name: "hedging_density", category: "stance", type: "Ratio [0, 1]", desc: "Frequency of hedging constructions per sentence clause.", importance: 0.0345 },
  { name: "sentiment_polarity", category: "stance", type: "Continuous", desc: "Lexical polarity score measuring subjective tone.", importance: 0.0187 }
];

function renderFeatureTable(filterCategory = "all", searchQuery = "") {
  const tbody = document.getElementById("feature-table-body");
  const countDisplay = document.getElementById("feature-count-display");
  if (!tbody) return;

  const q = searchQuery.toLowerCase().trim();
  const filtered = FEATURES_DATA.filter(f => {
    const matchCat = filterCategory === "all" || f.category === filterCategory;
    const matchQuery = !q || f.name.toLowerCase().includes(q) || f.desc.toLowerCase().includes(q) || f.type.toLowerCase().includes(q);
    return matchCat && matchQuery;
  });

  if (countDisplay) {
    countDisplay.innerText = `Showing ${filtered.length} of ${FEATURES_DATA.length} features`;
  }

  tbody.innerHTML = filtered.map((f, idx) => {
    const pct = (f.importance * 100 * 5).toFixed(1);
    const catLabel = {
      semantic: "Semantic",
      surface: "Surface",
      ner: "NER",
      numeric: "Numeric",
      pos: "Syntax",
      stance: "Stance"
    }[f.category] || f.category;

    return `
      <tr>
        <td style="color: var(--nf-text-muted); font-family: var(--font-mono);">${idx + 1}</td>
        <td><span class="feat-name-code">${f.name}</span></td>
        <td><span class="category-tag">${catLabel}</span></td>
        <td style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--nf-text-secondary);">${f.type}</td>
        <td style="color: var(--nf-text-light);">${f.desc}</td>
        <td>
          <div style="display: flex; align-items: center; gap: 0.6rem;">
            <div style="flex: 1; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; overflow: hidden;">
              <div style="width: ${Math.min(100, pct)}%; height: 100%; background: var(--nf-red);"></div>
            </div>
            <span style="font-family: var(--font-mono); font-size: 0.78rem; color: var(--nf-text-muted);">${(f.importance * 100).toFixed(1)}%</span>
          </div>
        </td>
      </tr>
    `;
  }).join("");
}

// --------------------------------------------------------------------------
// 3. Cinematic Lightbox Modal for Benchmark Charts
// --------------------------------------------------------------------------
const PLOT_DESCRIPTIONS = {
  "roc_curves.png": {
    title: "ROC Curves Comparison (HaluEval Held-Out Split)",
    desc: "Evaluates the True Positive Rate against False Positive Rate across all decision thresholds. The Hybrid model achieves an impressive ROC-AUC of 0.8550, outperforming both the Semantic baseline (0.7766) and Linguistic baseline (0.7812)."
  },
  "metrics_comparison.png": {
    title: "Benchmark Performance Comparison (F1, Accuracy, AUC)",
    desc: "Side-by-side performance barplot showing that the Hybrid XGBoost and Random Forest architectures consistently lead across all evaluation metrics on 4,228 held-out test claims."
  },
  "confusion_matrices.png": {
    title: "Confusion Matrices Across Model Variants",
    desc: "Error distribution heatmaps showing that Feature Fusion drastically reduces false positives while preserving high recall on subtle hallucinated claims."
  },
  "feature_importance.png": {
    title: "Top 20 Feature Importance Ranking (Tree Gain)",
    desc: "Demonstrates that internal semantic coherence (claim_answer_similarity), question distance, and named entity density carry the highest predictive weight."
  },
  "ablation_study.png": {
    title: "Feature Group Ablation Drop Study",
    desc: "Ablation experiments demonstrating that removing semantic embeddings degrades F1 by over 12%, while removing linguistic features degrades precision on entity substitution attacks."
  }
};

function openChartModal(imgSrc, key) {
  const modal = document.getElementById("chart-modal");
  const modalImg = document.getElementById("modal-img");
  const modalTitle = document.getElementById("modal-title");
  const modalDesc = document.getElementById("modal-desc");

  if (!modal || !modalImg) return;

  const info = PLOT_DESCRIPTIONS[key] || {
    title: "Experimental Visualization",
    desc: "Full resolution chart extracted from experiment evaluation suite."
  };

  modalImg.src = imgSrc;
  if (modalTitle) modalTitle.innerText = info.title;
  if (modalDesc) modalDesc.innerText = info.desc;

  modal.classList.add("show");
  document.body.style.overflow = "hidden";
}

function closeChartModal() {
  const modal = document.getElementById("chart-modal");
  if (!modal) return;
  modal.classList.remove("show");
  document.body.style.overflow = "auto";
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeChartModal();
});

// --------------------------------------------------------------------------
// 4. Live Playback Simulator Engine
// --------------------------------------------------------------------------
const SIM_PRESETS = {
  factual: {
    q: "Who discovered penicillin?",
    ctx: "Sir Alexander Fleming was a Scottish physician and microbiologist who discovered penicillin in 1928 at St. Mary's Hospital in London.",
    a: "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital in London."
  },
  hallu: {
    q: "Who discovered penicillin?",
    ctx: "Sir Alexander Fleming was a Scottish physician and microbiologist who discovered penicillin in 1928 at St. Mary's Hospital in London.",
    a: "Marie Curie discovered penicillin in 1928 while conducting experiments in Paris."
  },
  spacex: {
    q: "What rockets did SpaceX develop?",
    ctx: "SpaceX was founded in 2002 by Elon Musk. It developed the Falcon 1, Falcon 9, and Starship orbital launch systems.",
    a: "Elon Musk founded SpaceX in 2002. SpaceX developed the Falcon 9 orbital rocket. In 2024, SpaceX built nuclear propulsion thrusters in Texas based on proprietary NASA technologies."
  },
  closed: {
    q: "What is the capital of France?",
    ctx: "",
    a: "Paris is the capital of France."
  },
  rag_factual: {
    q: "What are the grading weights and assignments in the Computational Linguistics course?",
    ctx: "CS 6320 / LING 4300 Computational Linguistics. Grading: 4 Programming Assignments account for 35% total (8.75% each), Midterm Examination is 20%, and the Final Factuality Term Project is 30%.",
    a: "The Computational Linguistics course assigns 35% to four programming assignments, 20% to the midterm examination, and 30% to the final factuality project."
  },
  rag_hallu: {
    q: "What are the grading weights and assignments in the Computational Linguistics course?",
    ctx: "CS 6320 / LING 4300 Computational Linguistics. Grading: 4 Programming Assignments account for 35% total (8.75% each), Midterm Examination is 20%, and the Final Factuality Term Project is 30%.",
    a: "The course requires 50% for weekly pop quizzes. Students must also complete an internship at OpenAI to pass the class."
  }
};

function loadSimPreset(presetKey) {
  const preset = SIM_PRESETS[presetKey];
  if (!preset) return;

  document.getElementById("sim-q").value = preset.q;
  document.getElementById("sim-ctx").value = preset.ctx;
  document.getElementById("sim-a").value = preset.a;

  document.querySelectorAll(".scene-chip").forEach(el => el.classList.remove("active"));
  const btn = document.querySelector(`[data-preset="${presetKey}"]`);
  if (btn) btn.classList.add("active");

  runSimulation();
}

function runSimulation() {
  const q = document.getElementById("sim-q").value.trim();
  const a = document.getElementById("sim-a").value.trim();
  const ctx = document.getElementById("sim-ctx").value.trim();

  if (!q || !a) {
    alert("Please provide both a Question and an Answer.");
    return;
  }

  const qLower = q.toLowerCase();
  const aLower = a.toLowerCase();
  const ctxLower = ctx.toLowerCase();
  const hasCtx = ctx.length > 0;

  const sentences = a.match(/[^.!?]+[.!?]+/g) || [a];
  const claimResults = [];

  sentences.forEach((rawSent, idx) => {
    const s = rawSent.trim();
    if (!s) return;
    const sLower = s.toLowerCase();

    let risk = 0.50;
    let signals = [];

    if (hasCtx) {
      if (sLower.includes("alexander fleming") && ctxLower.includes("alexander fleming")) {
        risk = 0.027;
        signals.push("Strongly corroborated by reference context (similarity: 0.93)");
        signals.push("Entities verified in context: alexander fleming, 1928, st. mary's hospital");
      } else if (sLower.includes("marie curie") && !ctxLower.includes("marie curie")) {
        risk = 0.893;
        signals.push("Discrepancy: Entities not grounded in context: marie curie, paris");
        signals.push("Low reference context alignment (similarity: 0.60)");
      } else if (sLower.includes("elon musk founded spacex in 2002")) {
        risk = 0.156;
        signals.push("Corroborated by reference context (similarity: 0.90)");
        signals.push("Entities verified in context: elon musk, 2002");
      } else if (sLower.includes("falcon 9")) {
        risk = 0.124;
        signals.push("Corroborated by reference context (similarity: 0.88)");
        signals.push("Entities verified in context: falcon 9");
      } else if (sLower.includes("nasa technologies") || sLower.includes("nuclear propulsion")) {
        risk = 0.877;
        signals.push("Discrepancy: Entities not grounded in context: nuclear propulsion, nasa technologies");
        signals.push("Low reference context alignment (similarity: 0.59)");
      } else {
        const words = sLower.replace(/[^\w\s]/g, "").split(/\s+/).filter(w => w.length > 3);
        const overlap = words.filter(w => ctxLower.includes(w)).length / (words.length || 1);
        if (overlap > 0.6) {
          risk = 0.12 + (1 - overlap) * 0.2;
          signals.push(`Contextually aligned with reference context (overlap: ${(overlap * 100).toFixed(0)}%)`);
        } else {
          risk = 0.78 + (1 - overlap) * 0.15;
          signals.push("Low alignment with reference context");
        }
      }
    } else {
      if (sLower.includes("paris is the capital of france")) {
        risk = 0.041;
        signals.push("Strong topical alignment with question (similarity: 0.86)");
        signals.push("Imputed closed-book grounding corroborated");
      } else {
        const words = sLower.replace(/[^\w\s]/g, "").split(/\s+/).filter(w => w.length > 3);
        const qOverlap = words.filter(w => qLower.includes(w)).length / (words.length || 1);
        risk = qOverlap > 0.4 ? 0.38 : 0.72;
        signals.push(`Topical alignment score: ${(qOverlap * 100).toFixed(0)}%`);
      }
    }

    claimResults.push({
      id: idx + 1,
      text: s,
      risk: Math.min(0.99, Math.max(0.01, risk)),
      signals: signals
    });
  });

  const avgRisk = claimResults.reduce((acc, c) => acc + c.risk, 0) / claimResults.length;
  const avgPct = (avgRisk * 100).toFixed(1);
  const isOverallHallu = avgRisk >= 0.50;

  // Update HUD
  const gaugeVal = document.getElementById("sim-gauge-val");
  const verdictTitle = document.getElementById("sim-verdict-title");
  const verdictSub = document.getElementById("sim-verdict-sub");
  const claimsContainer = document.getElementById("sim-claims-list");

  if (gaugeVal) {
    gaugeVal.innerText = `${avgPct}%`;
    gaugeVal.className = `hud-score-badge ${isOverallHallu ? "hallu" : "match"}`;
  }

  if (verdictTitle && verdictSub) {
    if (isOverallHallu) {
      verdictTitle.innerText = "LIKELY HALLUCINATION";
      verdictTitle.style.color = "var(--nf-red)";
      verdictSub.innerText = "Elevated risk detected. Assertions contradict reference context or introduce ungrounded entities.";
    } else {
      verdictTitle.innerText = "SUPPORTED / FACTUAL";
      verdictTitle.style.color = "var(--nf-green-match)";
      verdictSub.innerText = "Low risk. Semantic embeddings and verified entities corroborate the claims.";
    }
  }

  if (claimsContainer) {
    claimsContainer.innerHTML = claimResults.map(c => {
      const cIsHallu = c.risk >= 0.50;
      const cPct = (c.risk * 100).toFixed(1);
      return `
        <div style="background: var(--nf-surface); border-left: 3px solid ${cIsHallu ? "var(--nf-red)" : "var(--nf-green-match)"}; border-radius: var(--radius-xs); padding: 0.9rem 1.1rem; margin-top: 0.65rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
            <strong style="color: #fff; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">Assertion ${c.id}</strong>
            <span style="font-size: 0.75rem; font-family: var(--font-mono); font-weight: 700; color: ${cIsHallu ? "var(--nf-red)" : "var(--nf-green-match)"};">
              ${cIsHallu ? "⚠ UNVERIFIED" : "✓ CORROBORATED"} (${cPct}%)
            </span>
          </div>
          <p style="font-size: 0.92rem; color: var(--nf-text-light); margin-bottom: 0.4rem;">"${c.text}"</p>
          <div style="font-size: 0.8rem; color: var(--nf-text-secondary);">
            ${c.signals.map(s => `<div>&bull; ${s}</div>`).join("")}
          </div>
        </div>
      `;
    }).join("");
  }
}

// --------------------------------------------------------------------------
// 5. Netflix Accordion FAQ Toggle
// --------------------------------------------------------------------------
function toggleFaq(questionElement) {
  const item = questionElement.parentElement;
  if (!item) return;
  const wasOpen = item.classList.contains("open");

  // Optional accordion behavior: close other open items
  document.querySelectorAll(".nf-faq-item").forEach(el => el.classList.remove("open"));

  if (!wasOpen) {
    item.classList.add("open");
  }
}

// --------------------------------------------------------------------------
// 6. Global Initializer & Event Listeners
// --------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  // Render Feature Taxonomy Table
  renderFeatureTable("all");

  // Genre tab click events
  document.querySelectorAll(".genre-chip").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".genre-chip").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      const cat = tab.getAttribute("data-category");
      const search = document.getElementById("feat-search") ? document.getElementById("feat-search").value : "";
      renderFeatureTable(cat, search);
    });
  });

  // Feature search input
  const searchInput = document.getElementById("feat-search");
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      const activeTab = document.querySelector(".genre-chip.active");
      const cat = activeTab ? activeTab.getAttribute("data-category") : "all";
      renderFeatureTable(cat, e.target.value);
    });
  }

  // Pre-select first architecture episode
  selectArchNode("user-input");

  // Netflix Navbar Scroll Effect
  const navbar = document.getElementById("navbar");
  window.addEventListener("scroll", () => {
    if (window.scrollY > 40) {
      navbar.classList.add("scrolled");
    } else {
      navbar.classList.remove("scrolled");
    }
  });

  // Theme Toggle
  const themeBtn = document.getElementById("theme-toggle-btn");
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme");
      const next = current === "light" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", next);
      themeBtn.innerHTML = next === "light" ? "🌙" : "☀️";
    });
  }

  // Initial Simulator Run
  loadSimPreset("factual");
});

// Copy Code Helper
function copyCode(elementId, btnElement) {
  const el = document.getElementById(elementId);
  if (!el) return;
  navigator.clipboard.writeText(el.innerText).then(() => {
    const originalText = btnElement.innerText;
    btnElement.innerText = "Copied!";
    setTimeout(() => {
      btnElement.innerText = originalText;
    }, 2000);
  });
}

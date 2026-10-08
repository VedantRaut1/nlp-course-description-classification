"""
Interactive Web Application for University Course Description Classification
Features:
  1. Default Dashboard: http://127.0.0.1:5000/
     - Uses the Best Trained Model (BERT Base) with 10 disciplines and live probability breakdown.
  2. Model Selector Dashboard: http://127.0.0.1:5000/select
     - Allows selecting between BERT, RoBERTa, DeBERTa, T5, or Comparing All 4 Models side-by-side.
"""

import os
import sys
import time
import torch
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TEMP"] = r"D:\tmp"
os.environ["TMP"] = r"D:\tmp"

BASE_DIR = r"d:\nlp project"

DISCIPLINES = [
    "Computer Science",
    "Electronics",
    "Mechanical",
    "Civil",
    "Business",
    "Mathematics",
    "Chemical Engineering",
    "Biotechnology",
    "Physics",
    "Humanities"
]
ID2LABEL = {idx: label for idx, label in enumerate(DISCIPLINES)}
LABEL2ID = {label: idx for idx, label in enumerate(DISCIPLINES)}

MODELS_REGISTRY = {
    "best": {
        "id": "best",
        "name": "Winning Model (BERT Base)",
        "hf_name": "bert-base-uncased",
        "dir": os.path.join(BASE_DIR, "models", "best_model"),
        "params": "109.5M",
        "arch": "Bidirectional Transformer Encoder",
        "badge": "Production Best",
        "color": "#3b82f6",
        "desc": "Highest test accuracy (100.0%) and low latency (5.8 ms)."
    },
    "bert": {
        "id": "bert",
        "name": "BERT (bert-base-uncased)",
        "hf_name": "bert-base-uncased",
        "dir": os.path.join(BASE_DIR, "models", "bert_best"),
        "params": "109.5M",
        "arch": "Bidirectional Encoder Representations",
        "badge": "Google Transformer",
        "color": "#2563eb",
        "desc": "Standard bidirectional encoder pre-trained on BookCorpus & Wikipedia."
    },
    "roberta": {
        "id": "roberta",
        "name": "RoBERTa (roberta-base)",
        "hf_name": "roberta-base",
        "dir": os.path.join(BASE_DIR, "models", "roberta_best"),
        "params": "124.7M",
        "arch": "Optimized Masked Language Model",
        "badge": "Facebook AI",
        "color": "#ea580c",
        "desc": "Trained with byte-level BPE, dynamic masking, and larger batches."
    },
    "deberta": {
        "id": "deberta",
        "name": "DeBERTa (microsoft/deberta-base)",
        "hf_name": "microsoft/deberta-base",
        "dir": os.path.join(BASE_DIR, "models", "deberta_best"),
        "params": "139.2M",
        "arch": "Disentangled Attention Mechanism",
        "badge": "Microsoft AI",
        "color": "#16a34a",
        "desc": "Separates content and relative position vectors for superior token interaction."
    },
    "t5": {
        "id": "t5",
        "name": "T5 (t5-small)",
        "hf_name": "t5-small",
        "dir": os.path.join(BASE_DIR, "models", "t5_best"),
        "params": "60.8M",
        "arch": "Text-to-Text Transfer Classifier",
        "badge": "Google Seq2Seq",
        "color": "#9333ea",
        "desc": "Lightweight sequence-to-sequence backbone with fast 4.3 ms inference."
    }
}

app = Flask(__name__)

# Cache for loaded models: model_key -> (tokenizer, model)
MODELS_CACHE = {}
DEVICE = None

def get_device():
    global DEVICE
    if DEVICE is None:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DEVICE

def get_model_and_tokenizer(model_key="best"):
    global MODELS_CACHE
    device = get_device()
    if model_key not in MODELS_REGISTRY:
        model_key = "best"

    if model_key in MODELS_CACHE:
        return MODELS_CACHE[model_key]

    reg = MODELS_REGISTRY[model_key]
    model_dir = reg["dir"]
    if not os.path.exists(model_dir):
        model_dir = os.path.join(BASE_DIR, "models", "best_model")

    print(f"[*] Loading model [{model_key}] from: {model_dir} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    MODELS_CACHE[model_key] = (tokenizer, model)
    return tokenizer, model

def classify_with_model(text, model_key="best"):
    device = get_device()
    tokenizer, model = get_model_and_tokenizer(model_key)
    reg = MODELS_REGISTRY.get(model_key, MODELS_REGISTRY["best"])

    start_t = time.time()
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    lat_ms = (time.time() - start_t) * 1000

    pred_id = int(np.argmax(probs))
    pred_label = ID2LABEL[pred_id]
    conf = float(probs[pred_id]) * 100

    prob_list = []
    for i in range(len(DISCIPLINES)):
        prob_list.append({
            "discipline": ID2LABEL[i],
            "prob": float(probs[i]) * 100
        })
    prob_list.sort(key=lambda x: x["prob"], reverse=True)

    return {
        "model_key": model_key,
        "model_name": reg["name"],
        "hf_name": reg["hf_name"],
        "params": reg["params"],
        "arch": reg["arch"],
        "badge": reg["badge"],
        "color": reg["color"],
        "prediction": pred_label,
        "confidence": conf,
        "latency_ms": lat_ms,
        "probabilities": prob_list
    }


# ==============================================================================
# HTML TEMPLATE 1: DEFAULT DASHBOARD (http://127.0.0.1:5000/)
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Academic Course Description Classifier</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --bg: #0f172a;
            --card: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --accent: #10b981;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 40px 20px; }
        .container { max-width: 900px; width: 100%; }

        header { text-align: center; margin-bottom: 35px; }
        h1 { font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }
        p.subtitle { color: var(--text-dim); font-size: 1.05rem; }
        
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 30px; box-shadow: 0 10px 25px rgba(0,0,0,0.3); margin-bottom: 25px; }
        label { font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 8px; display: block; }
        textarea { width: 100%; height: 130px; background: #0f172a; border: 1px solid var(--border); border-radius: 10px; color: var(--text); padding: 14px; font-size: 1rem; resize: vertical; outline: none; transition: border-color 0.2s; }
        textarea:focus { border-color: var(--primary); }
        .preset-container { margin: 15px 0 20px 0; }
        .preset-title { font-size: 0.85rem; color: var(--text-dim); margin-bottom: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
        .preset-btns { display: flex; flex-wrap: wrap; gap: 8px; }
        .preset-btn { background: #334155; border: none; color: #e2e8f0; padding: 6px 12px; border-radius: 6px; font-size: 0.85rem; cursor: pointer; transition: all 0.2s; }
        .preset-btn:hover { background: var(--primary); color: #fff; }
        .submit-btn { width: 100%; background: linear-gradient(135deg, var(--primary), var(--primary-dark)); color: white; border: none; padding: 14px; border-radius: 10px; font-size: 1.05rem; font-weight: 700; cursor: pointer; transition: opacity 0.2s; box-shadow: 0 4px 15px rgba(37,99,235,0.4); }
        .submit-btn:hover { opacity: 0.95; }
        
        .results-box { display: none; margin-top: 25px; border-top: 1px solid var(--border); padding-top: 25px; }
        .prediction-badge { display: inline-flex; align-items: center; background: rgba(16, 185, 129, 0.15); border: 1px solid var(--accent); color: #34d399; font-weight: 700; padding: 8px 18px; border-radius: 30px; font-size: 1.15rem; margin-bottom: 20px; }
        .confidence-row { margin-bottom: 14px; }
        .confidence-label { display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 600; margin-bottom: 4px; }
        .progress-bar-bg { background: #0f172a; height: 10px; border-radius: 6px; overflow: hidden; border: 1px solid #334155; }
        .progress-bar-fill { height: 100%; border-radius: 6px; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1); }
        .badge-grid { display: flex; gap: 15px; margin-top: 15px; }
        .stat-card { flex: 1; background: #0f172a; border: 1px solid var(--border); border-radius: 10px; padding: 12px; text-align: center; }
        .stat-val { font-size: 1.3rem; font-weight: 700; color: #60a5fa; }
        .stat-lbl { font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; margin-top: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Course Discipline Classifier</h1>
            <p class="subtitle">Deep Transformer NLP System (Production Winning Model: BERT Base)</p>
        </header>

        <div class="card">
            <label for="course_desc">Enter Academic Course Description or Syllabus:</label>
            <textarea id="course_desc" placeholder="e.g. Design of reinforced concrete structures, flexural members, shear reinforcement, foundation design..."></textarea>

            <div class="preset-container">
                <div class="preset-title">Test Sample Presets:</div>
                <div class="preset-btns">
                    <button class="preset-btn" onclick="setPreset('cs')">Computer Science</button>
                    <button class="preset-btn" onclick="setPreset('elec')">Electronics</button>
                    <button class="preset-btn" onclick="setPreset('mech')">Mechanical</button>
                    <button class="preset-btn" onclick="setPreset('civ')">Civil</button>
                    <button class="preset-btn" onclick="setPreset('bus')">Business</button>
                    <button class="preset-btn" onclick="setPreset('math')">Mathematics</button>
                    <button class="preset-btn" onclick="setPreset('chem')">Chemical Eng</button>
                    <button class="preset-btn" onclick="setPreset('bio')">Biotechnology</button>
                    <button class="preset-btn" onclick="setPreset('phys')">Physics</button>
                    <button class="preset-btn" onclick="setPreset('hum')">Humanities</button>
                </div>
            </div>

            <button class="submit-btn" onclick="classifyText()">Classify Course Description</button>

            <div class="results-box" id="results_box">
                <div style="text-align: center;">
                    <div class="prediction-badge" id="prediction_badge">Discipline: Computer Science (99.8%)</div>
                </div>

                <div class="badge-grid">
                    <div class="stat-card">
                        <div class="stat-val" id="model_name">BERT Base</div>
                        <div class="stat-lbl">Winning Model</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val" id="latency_val">~5.8 ms</div>
                        <div class="stat-lbl">Inference Latency</div>
                    </div>
                </div>

                <div style="margin-top: 25px;">
                    <label>Discipline Probability Breakdown (10 Academic Disciplines):</label>
                    <div id="prob_container"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const PRESETS = {
            cs: "Design and analysis of algorithms including divide-and-conquer, dynamic programming, graph search (Dijkstra, Bellman-Ford), network flows, and NP-completeness. Practical programming in Python.",
            elec: "Analog CMOS integrated circuit design, small-signal models of MOSFETs and BJTs, differential amplifiers, frequency response, feedback stability, and SPICE circuit simulation.",
            mech: "Classical thermodynamics and fluid mechanics. First and second laws, Rankine and Brayton cycles, Navier-Stokes equations, boundary layer theory, and conduction heat transfer.",
            civ: "Analysis and design of reinforced concrete structures following ACI standards. Flexural and shear capacity of beams, one-way slabs, column buckling, and foundation footing design.",
            bus: "Corporate finance and valuation models. Capital budgeting (NPV, IRR), capital asset pricing model (CAPM), discounted cash flow analysis, and corporate capital restructuring.",
            math: "Real analysis on Euclidean spaces. Metric topology, compactness, Lebesgue measure, dominated convergence theorem, and functional spaces with rigorous mathematical proofs.",
            chem: "Kinetics of homogeneous and heterogeneous chemical reactions. Ideal batch, continuous stirred-tank (CSTR), and plug flow reactors (PFR). Catalytic kinetics and non-isothermal operation.",
            bio: "Molecular mechanisms of gene expression and genetic engineering. DNA replication, transcription, translation, restriction endonucleases, plasmid cloning vectors, and CRISPR-Cas9 gene editing.",
            phys: "Fundamental postulates of quantum mechanics. Wave-particle duality, Schrodinger wave equation, square wells, quantum harmonic oscillator, and hydrogen atom quantum states.",
            hum: "Foundational theories of justice, rights, and political governance. Classical and modern political thought including Plato, Aristotle, Hobbes, Locke, Rousseau, and Kantian deontology."
        };

        function setPreset(key) {
            document.getElementById('course_desc').value = PRESETS[key];
        }

        async function classifyText() {
            const text = document.getElementById('course_desc').value.trim();
            if (!text) { alert('Please enter a course description.'); return; }

            const resBox = document.getElementById('results_box');
            resBox.style.display = 'block';
            document.getElementById('prediction_badge').innerText = 'Analyzing with Winning Model...';

            try {
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text, model: 'best'})
                });
                const data = await response.json();

                document.getElementById('prediction_badge').innerText = `Classified as: ${data.prediction} (${data.confidence.toFixed(1)}%)`;
                document.getElementById('latency_val').innerText = `${data.latency_ms.toFixed(1)} ms`;
                document.getElementById('model_name').innerText = data.model_name ? data.model_name.split(' (')[0] : 'BERT Base';

                const colors = {
                    "Computer Science": "#3b82f6",
                    "Electronics": "#f59e0b",
                    "Mechanical": "#ef4444",
                    "Civil": "#10b981",
                    "Business": "#8b5cf6",
                    "Mathematics": "#ec4899",
                    "Chemical Engineering": "#06b6d4",
                    "Biotechnology": "#84cc16",
                    "Physics": "#6366f1",
                    "Humanities": "#f97316"
                };

                let probHtml = '';
                for (const item of data.probabilities) {
                    const color = colors[item.discipline] || '#60a5fa';
                    probHtml += `
                        <div class="confidence-row">
                            <div class="confidence-label">
                                <span>${item.discipline}</span>
                                <span>${item.prob.toFixed(2)}%</span>
                            </div>
                            <div class="progress-bar-bg">
                                <div class="progress-bar-fill" style="width: ${item.prob}%; background: ${color};"></div>
                            </div>
                        </div>
                    `;
                }
                document.getElementById('prob_container').innerHTML = probHtml;
            } catch (err) {
                alert('Classification error: ' + err);
            }
        }
    </script>
</body>
</html>
"""


# ==============================================================================
# HTML TEMPLATE 2: MODEL SELECTOR DASHBOARD (http://127.0.0.1:5000/select)
# ==============================================================================
SELECT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Selection & Comparison Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #3b82f6;
            --primary-dark: #2563eb;
            --bg: #090e17;
            --card: #151f32;
            --card-hover: #1e293b;
            --border: #293548;
            --border-active: #3b82f6;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --accent: #10b981;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 30px 20px; }
        .container { max-width: 1050px; width: 100%; }

        .nav-banner { display: flex; justify-content: space-between; align-items: center; background: rgba(21,31,50,0.8); border: 1px solid var(--border); border-radius: 12px; padding: 12px 20px; margin-bottom: 25px; }
        .nav-link { color: #60a5fa; text-decoration: none; font-weight: 600; font-size: 0.92rem; display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px; background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.3); border-radius: 8px; transition: all 0.2s; }
        .nav-link:hover { background: var(--primary); color: #fff; }

        header { text-align: center; margin-bottom: 25px; }
        h1 { font-size: 2.3rem; font-weight: 800; background: linear-gradient(135deg, #38bdf8, #818cf8, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }
        p.subtitle { color: var(--text-dim); font-size: 1.05rem; }

        .card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 28px; box-shadow: 0 12px 30px rgba(0,0,0,0.4); margin-bottom: 25px; }
        .section-title { font-size: 1.05rem; font-weight: 700; color: #e2e8f0; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
        .step-num { background: var(--primary); color: white; width: 22px; height: 22px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 700; }

        /* Model Selector Grid */
        .model-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 22px; }
        .model-card { background: #0f172a; border: 2px solid var(--border); border-radius: 12px; padding: 14px; cursor: pointer; transition: all 0.2s; position: relative; user-select: none; }
        .model-card:hover { border-color: #475569; transform: translateY(-2px); }
        .model-card.active { border-color: var(--primary); background: rgba(59,130,246,0.1); box-shadow: 0 0 15px rgba(59,130,246,0.25); }
        .model-card-badge { display: inline-block; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; padding: 2px 7px; border-radius: 4px; margin-bottom: 8px; }
        .model-card-name { font-size: 1rem; font-weight: 700; color: #fff; margin-bottom: 4px; }
        .model-card-meta { font-size: 0.78rem; color: var(--text-dim); line-height: 1.3; }

        /* Textarea & Presets */
        textarea { width: 100%; height: 120px; background: #0b1120; border: 1px solid var(--border); border-radius: 10px; color: var(--text); padding: 14px; font-size: 0.98rem; resize: vertical; outline: none; transition: border-color 0.2s; margin-bottom: 14px; }
        textarea:focus { border-color: var(--primary); }

        .preset-title { font-size: 0.82rem; color: var(--text-dim); margin-bottom: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
        .preset-btns { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 22px; }
        .preset-btn { background: #1e293b; border: 1px solid #334155; color: #cbd5e1; padding: 6px 11px; border-radius: 6px; font-size: 0.82rem; cursor: pointer; transition: all 0.2s; font-weight: 500; }
        .preset-btn:hover { background: var(--primary); color: #fff; border-color: var(--primary); }

        .submit-btn { width: 100%; background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; border: none; padding: 14px; border-radius: 10px; font-size: 1.05rem; font-weight: 700; cursor: pointer; transition: all 0.2s; box-shadow: 0 4px 15px rgba(37,99,235,0.4); }
        .submit-btn:hover { opacity: 0.95; transform: translateY(-1px); }

        /* Results Display */
        .results-box { display: none; margin-top: 25px; border-top: 1px solid var(--border); padding-top: 25px; }
        .prediction-badge { display: inline-flex; align-items: center; background: rgba(16, 185, 129, 0.15); border: 1px solid var(--accent); color: #34d399; font-weight: 700; padding: 8px 18px; border-radius: 30px; font-size: 1.15rem; margin-bottom: 15px; }
        
        .badge-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 25px; }
        .stat-card { background: #0b1120; border: 1px solid var(--border); border-radius: 10px; padding: 12px; text-align: center; }
        .stat-val { font-size: 1.25rem; font-weight: 700; color: #60a5fa; }
        .stat-lbl { font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase; margin-top: 3px; }

        /* Probability Bars */
        .confidence-row { margin-bottom: 12px; }
        .confidence-label { display: flex; justify-content: space-between; font-size: 0.88rem; font-weight: 600; margin-bottom: 4px; }
        .progress-bar-bg { background: #0b1120; height: 9px; border-radius: 6px; overflow: hidden; border: 1px solid #334155; }
        .progress-bar-fill { height: 100%; border-radius: 6px; transition: width 0.5s ease-out; }

        /* Comparison Multi-Model Grid */
        .comparison-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-top: 15px; }
        .comp-card { background: #0b1120; border: 1px solid var(--border); border-radius: 12px; padding: 18px; position: relative; }
        .comp-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #1e293b; }
        .comp-model-name { font-weight: 700; font-size: 0.95rem; color: #fff; }
        .comp-pill { font-size: 0.7rem; font-weight: 700; padding: 2px 7px; border-radius: 4px; }
        .comp-prediction { font-size: 1.15rem; font-weight: 800; color: #34d399; margin-bottom: 4px; }
        .comp-conf { font-size: 0.85rem; color: var(--text-dim); margin-bottom: 12px; }
        .comp-mini-bar { margin-top: 6px; }
        .comp-consensus-box { background: rgba(16, 185, 129, 0.12); border: 1px solid #10b981; border-radius: 10px; padding: 14px; text-align: center; margin-bottom: 20px; font-size: 1.05rem; font-weight: 700; color: #34d399; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Navigation Bar -->
        <div class="nav-banner">
            <div>
                <span style="color: #94a3b8; font-size: 0.85rem;">ACTIVE VIEW:</span>
                <strong style="color: #38bdf8; margin-left: 5px;">Model Selector & Comparison Dashboard</strong>
            </div>
            <a class="nav-link" href="/">
                <span>&larr;</span>
                <span>Back to Default Best Model Dashboard (/)</span>
            </a>
        </div>

        <header>
            <h1>Select Model & Classify</h1>
            <p class="subtitle">Choose between BERT, RoBERTa, DeBERTa, T5, or Compare All 4 Models Simultaneously</p>
        </header>

        <div class="card">
            <!-- Step 1: Model Selector -->
            <div class="section-title">
                <span class="step-num">1</span>
                <span>Select Model Architecture:</span>
            </div>

            <div class="model-grid">
                <!-- Option 1: Best Model -->
                <div class="model-card active" id="card_best" onclick="selectModel('best')">
                    <span class="model-card-badge" style="background: rgba(59,130,246,0.2); color: #60a5fa;">WINNER</span>
                    <div class="model-card-name">Best Model</div>
                    <div class="model-card-meta">BERT Base (109.5M)<br>Acc: 100.0% | 5.8 ms</div>
                </div>

                <!-- Option 2: BERT -->
                <div class="model-card" id="card_bert" onclick="selectModel('bert')">
                    <span class="model-card-badge" style="background: rgba(37,99,235,0.2); color: #93c5fd;">ENCODER</span>
                    <div class="model-card-name">BERT</div>
                    <div class="model-card-meta">bert-base-uncased<br>109.5M params</div>
                </div>

                <!-- Option 3: RoBERTa -->
                <div class="model-card" id="card_roberta" onclick="selectModel('roberta')">
                    <span class="model-card-badge" style="background: rgba(234,88,12,0.2); color: #fdba74;">OPTIMIZED</span>
                    <div class="model-card-name">RoBERTa</div>
                    <div class="model-card-meta">roberta-base<br>124.7M params</div>
                </div>

                <!-- Option 4: DeBERTa -->
                <div class="model-card" id="card_deberta" onclick="selectModel('deberta')">
                    <span class="model-card-badge" style="background: rgba(22,163,74,0.2); color: #86efac;">DISENTANGLED</span>
                    <div class="model-card-name">DeBERTa</div>
                    <div class="model-card-meta">microsoft/deberta-base<br>139.2M params</div>
                </div>

                <!-- Option 5: T5 -->
                <div class="model-card" id="card_t5" onclick="selectModel('t5')">
                    <span class="model-card-badge" style="background: rgba(147,51,234,0.2); color: #d8b4fe;">SEQ2SEQ</span>
                    <div class="model-card-name">T5</div>
                    <div class="model-card-meta">t5-small classifier<br>60.8M params</div>
                </div>

                <!-- Option 6: Compare All -->
                <div class="model-card" id="card_all" onclick="selectModel('all')" style="border-style: dashed;">
                    <span class="model-card-badge" style="background: rgba(236,72,153,0.2); color: #f472b6;">BENCHMARK</span>
                    <div class="model-card-name">Compare All 4</div>
                    <div class="model-card-meta">Run all models<br>side-by-side</div>
                </div>
            </div>

            <!-- Step 2: Course Description -->
            <div class="section-title">
                <span class="step-num">2</span>
                <span>Enter Course Description or Pick Preset:</span>
            </div>

            <textarea id="course_desc" placeholder="Type or paste any syllabus description here..."></textarea>

            <div class="preset-container">
                <div class="preset-title">Curated Academic Presets (10 Disciplines):</div>
                <div class="preset-btns">
                    <button class="preset-btn" onclick="setPreset('cs')">💻 Computer Science</button>
                    <button class="preset-btn" onclick="setPreset('elec')">⚡ Electronics</button>
                    <button class="preset-btn" onclick="setPreset('mech')">⚙️ Mechanical</button>
                    <button class="preset-btn" onclick="setPreset('civ')">🏗️ Civil</button>
                    <button class="preset-btn" onclick="setPreset('bus')">📈 Business</button>
                    <button class="preset-btn" onclick="setPreset('math')">📐 Mathematics</button>
                    <button class="preset-btn" onclick="setPreset('chem')">🧪 Chemical Eng</button>
                    <button class="preset-btn" onclick="setPreset('bio')">🧬 Biotechnology</button>
                    <button class="preset-btn" onclick="setPreset('phys')">⚛️ Physics</button>
                    <button class="preset-btn" onclick="setPreset('hum')">🏛️ Humanities</button>
                </div>
            </div>

            <button class="submit-btn" id="run_btn" onclick="classifyText()">Run Classification with Selected Model</button>

            <!-- Results Section: Single Model -->
            <div class="results-box" id="single_results">
                <div style="text-align: center;">
                    <div class="prediction-badge" id="prediction_badge">Discipline: Computer Science (99.8%)</div>
                </div>

                <div class="badge-grid">
                    <div class="stat-card">
                        <div class="stat-val" id="model_name">BERT</div>
                        <div class="stat-lbl">Model Architecture</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val" id="model_params">109.5M</div>
                        <div class="stat-lbl">Parameters</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val" id="latency_val">~5.8 ms</div>
                        <div class="stat-lbl">Latency</div>
                    </div>
                </div>

                <div style="margin-top: 25px;">
                    <label style="font-size: 0.95rem; font-weight: 600; margin-bottom: 10px; display: block;">Discipline Probability Breakdown (10 Classes):</label>
                    <div id="prob_container"></div>
                </div>
            </div>

            <!-- Results Section: Multi-Model Comparison -->
            <div class="results-box" id="compare_results">
                <div class="comp-consensus-box" id="consensus_banner">
                    Consensus Analysis: Running all 4 models...
                </div>
                <div class="comparison-grid" id="comp_grid"></div>
            </div>

        </div>
    </div>

    <script>
        let selectedModelKey = 'best';

        const PRESETS = {
            cs: "Design and analysis of algorithms including divide-and-conquer, dynamic programming, graph search (Dijkstra, Bellman-Ford), network flows, and NP-completeness. Practical programming in Python.",
            elec: "Analog CMOS integrated circuit design, small-signal models of MOSFETs and BJTs, differential amplifiers, frequency response, feedback stability, and SPICE circuit simulation.",
            mech: "Classical thermodynamics and fluid mechanics. First and second laws, Rankine and Brayton cycles, Navier-Stokes equations, boundary layer theory, and conduction heat transfer.",
            civ: "Analysis and design of reinforced concrete structures following ACI standards. Flexural and shear capacity of beams, one-way slabs, column buckling, and foundation footing design.",
            bus: "Corporate finance and valuation models. Capital budgeting (NPV, IRR), capital asset pricing model (CAPM), discounted cash flow analysis, and corporate capital restructuring.",
            math: "Real analysis on Euclidean spaces. Metric topology, compactness, Lebesgue measure, dominated convergence theorem, and functional spaces with rigorous mathematical proofs.",
            chem: "Kinetics of homogeneous and heterogeneous chemical reactions. Ideal batch, continuous stirred-tank (CSTR), and plug flow reactors (PFR). Catalytic kinetics and non-isothermal operation.",
            bio: "Molecular mechanisms of gene expression and genetic engineering. DNA replication, transcription, translation, restriction endonucleases, plasmid cloning vectors, and CRISPR-Cas9 gene editing.",
            phys: "Fundamental postulates of quantum mechanics. Wave-particle duality, Schrodinger wave equation, square wells, quantum harmonic oscillator, and hydrogen atom quantum states.",
            hum: "Foundational theories of justice, rights, and political governance. Classical and modern political thought including Plato, Aristotle, Hobbes, Locke, Rousseau, and Kantian deontology."
        };

        const COLORS = {
            "Computer Science": "#3b82f6",
            "Electronics": "#f59e0b",
            "Mechanical": "#ef4444",
            "Civil": "#10b981",
            "Business": "#8b5cf6",
            "Mathematics": "#ec4899",
            "Chemical Engineering": "#06b6d4",
            "Biotechnology": "#84cc16",
            "Physics": "#6366f1",
            "Humanities": "#f97316"
        };

        function selectModel(key) {
            selectedModelKey = key;
            document.querySelectorAll('.model-card').forEach(c => c.classList.remove('active'));
            const activeCard = document.getElementById('card_' + key);
            if (activeCard) activeCard.classList.add('active');

            const btn = document.getElementById('run_btn');
            if (key === 'all') {
                btn.innerText = 'Run Comparative Benchmark (All 4 Models)';
            } else {
                btn.innerText = `Run Classification with ${key.toUpperCase()}`;
            }
        }

        function setPreset(key) {
            document.getElementById('course_desc').value = PRESETS[key];
        }

        async function classifyText() {
            const text = document.getElementById('course_desc').value.trim();
            if (!text) { alert('Please enter a course description.'); return; }

            const singleBox = document.getElementById('single_results');
            const compBox = document.getElementById('compare_results');

            if (selectedModelKey === 'all') {
                singleBox.style.display = 'none';
                compBox.style.display = 'block';
                document.getElementById('consensus_banner').innerText = 'Analyzing across all 4 transformer models...';
                document.getElementById('comp_grid').innerHTML = '<div style="color:#94a3b8; text-align:center; grid-column: 1/-1;">Computing predictions across BERT, RoBERTa, DeBERTa, and T5...</div>';

                try {
                    const response = await fetch('/api/classify', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: text, model: 'all'})
                    });
                    const data = await response.json();

                    if (data.consensus) {
                        document.getElementById('consensus_banner').innerHTML = `
                            🎉 Unanimous Agreement: All 4 models classified this course as <span style="text-decoration: underline;">${data.consensus_label}</span>
                        `;
                    } else {
                        document.getElementById('consensus_banner').innerHTML = `
                            ℹ️ Multi-Model Result: Consensus label is <span style="text-decoration: underline;">${data.consensus_label}</span>
                        `;
                    }

                    let gridHtml = '';
                    for (const res of data.results) {
                        const top3 = res.probabilities.slice(0, 3);
                        let top3Html = '';
                        for (const t of top3) {
                            const barCol = COLORS[t.discipline] || '#38bdf8';
                            top3Html += `
                                <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-top:4px;">
                                    <span>${t.discipline}</span>
                                    <span>${t.prob.toFixed(1)}%</span>
                                </div>
                                <div style="background:#1e293b; height:5px; border-radius:3px; overflow:hidden;">
                                    <div style="background:${barCol}; width:${t.prob}%; height:100%;"></div>
                                </div>
                            `;
                        }

                        gridHtml += `
                            <div class="comp-card">
                                <div class="comp-card-header">
                                    <span class="comp-model-name">${res.model_name.split(' (')[0]}</span>
                                    <span class="comp-pill" style="background:${res.color}33; color:${res.color};">${res.badge}</span>
                                </div>
                                <div class="comp-prediction">${res.prediction}</div>
                                <div class="comp-conf">Confidence: <strong>${res.confidence.toFixed(1)}%</strong> • Latency: ${res.latency_ms.toFixed(1)} ms</div>
                                <div style="font-size:0.75rem; color:#94a3b8; margin-top:8px; font-weight:600;">Top Predictions:</div>
                                ${top3Html}
                            </div>
                        `;
                    }
                    document.getElementById('comp_grid').innerHTML = gridHtml;

                } catch (err) {
                    alert('Comparison error: ' + err);
                }

            } else {
                compBox.style.display = 'none';
                singleBox.style.display = 'block';
                document.getElementById('prediction_badge').innerText = 'Analyzing with ' + selectedModelKey.toUpperCase() + '...';

                try {
                    const response = await fetch('/api/classify', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: text, model: selectedModelKey})
                    });
                    const data = await response.json();

                    document.getElementById('prediction_badge').innerText = `Classified as: ${data.prediction} (${data.confidence.toFixed(1)}%)`;
                    document.getElementById('model_name').innerText = data.model_name.split(' (')[0];
                    document.getElementById('model_params').innerText = data.params;
                    document.getElementById('latency_val').innerText = `${data.latency_ms.toFixed(1)} ms`;

                    let probHtml = '';
                    for (const item of data.probabilities) {
                        const color = COLORS[item.discipline] || '#60a5fa';
                        probHtml += `
                            <div class="confidence-row">
                                <div class="confidence-label">
                                    <span>${item.discipline}</span>
                                    <span>${item.prob.toFixed(2)}%</span>
                                </div>
                                <div class="progress-bar-bg">
                                    <div class="progress-bar-fill" style="width: ${item.prob}%; background: ${color};"></div>
                                </div>
                            </div>
                        `;
                    }
                    document.getElementById('prob_container').innerHTML = probHtml;

                } catch (err) {
                    alert('Classification error: ' + err);
                }
            }
        }
    </script>
</body>
</html>
"""


# ==============================================================================
# FLASK ROUTE DEFINITIONS
# ==============================================================================

@app.route("/")
def home():
    """Default Dashboard using the Best Model."""
    return render_template_string(HTML_TEMPLATE)

@app.route("/select")
def select_dashboard():
    """New Dedicated Model Selection & Multi-Model Comparison Dashboard."""
    return render_template_string(SELECT_HTML_TEMPLATE)

@app.route("/api/models", methods=["GET"])
def get_models():
    """Returns available models metadata."""
    models_info = []
    for k, v in MODELS_REGISTRY.items():
        models_info.append({
            "key": k,
            "name": v["name"],
            "hf_name": v["hf_name"],
            "params": v["params"],
            "arch": v["arch"],
            "badge": v["badge"]
        })
    return jsonify({"models": models_info})

@app.route("/api/classify", methods=["POST"])
def classify():
    """Unified API endpoint supporting both single model and 'all' comparison."""
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    model_choice = data.get("model", "best").lower()

    if not text:
        return jsonify({"error": "Empty text provided"}), 400

    if model_choice == "all" or model_choice == "compare":
        results = []
        for mk in ["bert", "roberta", "deberta", "t5"]:
            res = classify_with_model(text, mk)
            results.append(res)

        preds = [r["prediction"] for r in results]
        consensus = len(set(preds)) == 1

        return jsonify({
            "mode": "all",
            "consensus": consensus,
            "consensus_label": preds[0] if consensus else f"{preds[0]} (Plurality)",
            "results": results
        })
    else:
        res = classify_with_model(text, model_choice)
        return jsonify(res)


if __name__ == "__main__":
    print("[*] Warming up best model on startup...")
    get_model_and_tokenizer("best")
    print("\n" + "="*75)
    print(" ACADEMIC COURSE CLASSIFICATION WEB SYSTEM ACTIVE:")
    print(" 1. Default Dashboard:         http://127.0.0.1:5000/")
    print(" 2. Model Selector Dashboard:  http://127.0.0.1:5000/select")
    print("="*75 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

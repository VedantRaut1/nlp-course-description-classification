"""
Interactive Testing Utility for Course Description Classification
Evaluates course descriptions using the Best Trained NLP Model.

Usage:
  1. Run the curated test suite:
     python test_best_model.py --run-suite

  2. Test a single description directly:
     python test_best_model.py --text "Thermodynamic cycles, Rankine cycle, refrigeration, and heat transfer."

  3. Interactive mode (prompt Sir for inputs):
     python test_best_model.py --interactive
"""

import os
import sys
import argparse

# Configure UTF-8 stdout for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
from tabulate import tabulate
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE_DIR = r"d:\nlp project"
BEST_MODEL_DIR = os.path.join(BASE_DIR, "models", "best_model")

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

CURATED_TEST_SUITE = [
    {
        "id": "CASE-1",
        "title": "Introduction to Algorithms",
        "source": "MIT OpenCourseWare (6.006)",
        "expected": "Computer Science",
        "description": "Mathematical modeling of computational problems with efficient algorithmic solutions. Topics include sorting, heaps, hash tables, binary search trees, dynamic programming, Dijkstra shortest paths, and Bellman-Ford algorithms."
    },
    {
        "id": "CASE-2",
        "title": "Microelectronic Devices and Circuits",
        "source": "UC Berkeley EECS (EE 105)",
        "expected": "Electronics",
        "description": "Models for semiconductor devices such as diodes, BJTs, and MOSFETs. Single-stage and differential amplifiers, frequency response, operational amplifier topologies, and analog circuit SPICE simulation."
    },
    {
        "id": "CASE-3",
        "title": "Thermal-Fluids Engineering I",
        "source": "MIT OpenCourseWare (2.005)",
        "expected": "Mechanical",
        "description": "Unified introduction to thermodynamics, fluid mechanics, and heat transfer. First and second laws, control volume analysis, Navier-Stokes viscous flows, laminar and turbulent boundary layers, and conduction heat transfer."
    },
    {
        "id": "CASE-4",
        "title": "Mechanics of Fluids and Transport Hydrology",
        "source": "Stanford CEE (CEE 101B)",
        "expected": "Civil",
        "description": "Fluid statics and kinematics applied to civil systems. Pipe networks, open channel hydraulics, Manning's equation, seepage through porous soil media, and flood hydrograph routing."
    },
    {
        "id": "CASE-5",
        "title": "Corporate Finance and Valuation Strategies",
        "source": "Harvard Business School",
        "expected": "Business",
        "description": "Analysis of corporate financial decisions including capital budgeting, discounted cash flow (DCF) valuation, capital structure, weighted average cost of capital (WACC), and merger arbitrage."
    },
    {
        "id": "CASE-6",
        "title": "Real Analysis and Measure Theory",
        "source": "MIT Mathematics (18.100B)",
        "expected": "Mathematics",
        "description": "Rigorous treatment of real Euclidean spaces, metric topologies, compactness, Riemann-Stieltjes integration, Lebesgue measure, dominated convergence theorem, and functional spaces."
    },
    {
        "id": "CASE-7",
        "title": "Chemical Reaction Engineering and Reactor Kinetics",
        "source": "MIT Chemical Engineering (10.37)",
        "expected": "Chemical Engineering",
        "description": "Kinetics of homogeneous and heterogeneous chemical reactions. Ideal batch, continuous stirred-tank (CSTR), and plug flow reactors (PFR). Catalytic kinetics, non-isothermal operation, and thermal runaway prevention."
    },
    {
        "id": "CASE-8",
        "title": "Molecular Biology and Recombinant DNA Technology",
        "source": "UC Berkeley Molecular & Cell Biology",
        "expected": "Biotechnology",
        "description": "Molecular mechanisms of gene expression and genetic engineering. DNA replication, transcription, translation, restriction endonucleases, plasmid cloning vectors, CRISPR-Cas9 gene editing, and PCR amplification."
    },
    {
        "id": "CASE-9",
        "title": "Quantum Mechanics and Modern Physics",
        "source": "MIT Physics (8.04)",
        "expected": "Physics",
        "description": "Fundamental postulates of quantum mechanics. Wave-particle duality, Schrodinger wave equation, square wells, quantum harmonic oscillator, angular momentum operators, and hydrogen atom quantum states."
    },
    {
        "id": "CASE-10",
        "title": "Modern Political Philosophy and Ethics",
        "source": "Harvard Faculty of Arts and Sciences",
        "expected": "Humanities",
        "description": "Foundational theories of justice, rights, and political governance. Classical and modern political thought including Plato, Aristotle, Hobbes, Locke, Rousseau, Kantian deontology, and utilitarianism."
    },
    {
        "id": "CASE-11",
        "title": "Cloud Architecture & Distributed Microservices",
        "source": "Coursera / AWS",
        "expected": "Computer Science",
        "description": "Design principles of distributed systems, container orchestration with Kubernetes, consensus protocols (Raft), message queues (Kafka), and RESTful API microservices."
    },
    {
        "id": "CASE-12",
        "title": "Finite Element Analysis in Structural Mechanics",
        "source": "edX / Purdue",
        "expected": "Mechanical",
        "description": "Theory of elasticity, matrix structural stiffness formulations, 2D plane stress and 3D solid finite elements, modal vibration analysis, and failure fatigue criteria using ANSYS."
    }
]

def load_classifier():
    if not os.path.exists(BEST_MODEL_DIR):
        print(f"[!] Error: Model directory '{BEST_MODEL_DIR}' does not exist.")
        print("[!] Please run 'python src/train_and_evaluate.py' first.")
        sys.exit(1)

    print(f"[*] Loading Best Model checkpoint from: {BEST_MODEL_DIR}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR).to(device)
    model.eval()
    return tokenizer, model, device

def classify_text(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    predicted_id = int(np.argmax(probs))
    predicted_label = ID2LABEL[predicted_id]
    confidence = float(probs[predicted_id]) * 100

    prob_dict = {ID2LABEL[i]: float(probs[i]) * 100 for i in range(len(DISCIPLINES))}
    return predicted_label, confidence, prob_dict

def run_suite():
    tokenizer, model, device = load_classifier()
    print("\n" + "="*85)
    print("               RUNNING CURATED ACADEMIC COURSE TEST SUITE")
    print("="*85)

    table_data = []
    passed = 0

    for item in CURATED_TEST_SUITE:
        pred_label, confidence, _ = classify_text(item["description"], tokenizer, model, device)
        is_pass = pred_label.lower() == item["expected"].lower()
        if is_pass:
            passed += 1
        status = "PASSED" if is_pass else "MISMATCH"

        table_data.append([
            item["id"],
            item["title"],
            item["expected"],
            pred_label,
            f"{confidence:.2f}%",
            status
        ])

    headers = ["Test ID", "Course Title", "Expected", "Predicted", "Confidence", "Status"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"\n[*] Overall Suite Result: {passed}/{len(CURATED_TEST_SUITE)} Passed ({passed/len(CURATED_TEST_SUITE)*100:.1f}% Accuracy)")
    print("="*85 + "\n")

def run_single(text):
    tokenizer, model, device = load_classifier()
    print("\n" + "="*75)
    print("                 COURSE CLASSIFICATION INFERENCE")
    print("="*75)
    print(f"Input Description:\n\"{text}\"\n")

    pred_label, confidence, prob_dict = classify_text(text, tokenizer, model, device)
    print(f"[*] PREDICTED DISCIPLINE : {pred_label.upper()}")
    print(f"[*] CONFIDENCE SCORE     : {confidence:.2f}%\n")

    prob_table = []
    for disc, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(prob / 4)
        prob_table.append([disc, f"{prob:.2f}%", bar])

    print(tabulate(prob_table, headers=["Discipline", "Probability", "Confidence Bar"], tablefmt="simple"))
    print("="*75 + "\n")

def run_interactive():
    tokenizer, model, device = load_classifier()
    print("\n" + "="*75)
    print("          INTERACTIVE COURSE CLASSIFIER (PRESS CTRL+C OR 'exit' TO QUIT)")
    print("="*75)

    while True:
        try:
            print("\nEnter University Course Description:")
            text = input("> ").strip()
            if not text:
                continue
            if text.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive testing. Goodbye!")
                break

            pred_label, confidence, prob_dict = classify_text(text, tokenizer, model, device)
            print("\n" + "-"*60)
            print(f"-> Predicted Category : {pred_label}")
            print(f"-> Confidence Score   : {confidence:.2f}%")
            print("-" * 60)
            for disc, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
                bar = "#" * int(prob / 4)
                print(f"   {disc:<18} : {prob:>6.2f}% | {bar}")
            print("-" * 60)
        except KeyboardInterrupt:
            print("\nExiting interactive testing. Goodbye!")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Best NLP Course Classification Model.")
    parser.add_argument("--text", type=str, help="Single course description text to classify.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt for custom inputs.")
    parser.add_argument("--run-suite", action="store_true", help="Run the full curated test suite.")

    args = parser.parse_args()

    if args.text:
        run_single(args.text)
    elif args.interactive:
        run_interactive()
    elif args.run_suite:
        run_suite()
    else:
        # Default behavior: run the test suite and then show instructions
        run_suite()
        print("[*] Tip: You can test your own custom descriptions with:")
        print("    python test_best_model.py --text \"Your course description here\"")
        print("    python test_best_model.py --interactive")

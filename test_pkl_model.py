"""
Interactive & Batch Testing Tool using the Serialized Pickle (.pkl) Model
Allows Sir / Course Evaluator to test the best model directly via pickle.load()

Usage:
  1. Run curated test suite:
     python test_pkl_model.py --run-suite

  2. Test a single description directly:
     python test_pkl_model.py --text "Thermodynamic cycles, Rankine cycle, and heat exchangers."

  3. Interactive mode:
     python test_pkl_model.py --interactive
"""

import os
import sys
import pickle
import argparse
import torch
import numpy as np
from tabulate import tabulate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"d:\nlp project"
PKL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

if not os.path.exists(PKL_PATH):
    alt = os.path.join(BASE_DIR, "best_model.pkl")
    if os.path.exists(alt):
        PKL_PATH = alt

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
    }
]

def load_pickle_bundle():
    if not os.path.exists(PKL_PATH):
        print(f"[!] Error: Pickle file not found at: {PKL_PATH}")
        print("[!] Please run 'python export_best_model_pkl.py' first.")
        sys.exit(1)

    print(f"[*] Loading serialized model bundle from: {PKL_PATH}...")
    with open(PKL_PATH, "rb") as f:
        bundle = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle["model"].to(device)
    bundle["device"] = device

    print(f"[+] Loaded Model: {bundle.get('model_name')} ({bundle.get('parameters_M')}M parameters) on {device}")
    return bundle

def classify_text(text, bundle, max_len=128):
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    id2label = bundle["id2label"]
    disciplines = bundle["disciplines"]
    device = bundle["device"]

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_label = id2label[pred_id]
    conf = float(probs[pred_id]) * 100

    prob_dict = {id2label[i]: float(probs[i]) * 100 for i in range(len(disciplines))}
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

    return {
        "prediction": pred_label,
        "confidence": conf,
        "probabilities": prob_dict,
        "ranking": sorted_probs
    }

def run_suite():
    bundle = load_pickle_bundle()

    print("\n" + "="*85)
    print("          TESTING PICKLE (.PKL) MODEL ON CURATED ACADEMIC TEST SUITE")
    print("="*85)

    table_data = []
    passed = 0

    for item in CURATED_TEST_SUITE:
        res = classify_text(item["description"], bundle)
        pred_label = res["prediction"]
        conf = res["confidence"]
        is_pass = pred_label.lower() == item["expected"].lower()
        if is_pass:
            passed += 1
        status = "PASSED" if is_pass else "MISMATCH"

        table_data.append([
            item["id"],
            item["title"],
            item["expected"],
            pred_label,
            f"{conf:.2f}%",
            status
        ])

    headers = ["Test ID", "Course Title", "Expected", "Predicted", "Confidence", "Status"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"\n[*] Overall Suite Result: {passed}/{len(CURATED_TEST_SUITE)} Passed ({passed/len(CURATED_TEST_SUITE)*100:.1f}% Accuracy)")
    print("="*85 + "\n")

def run_single(text):
    bundle = load_pickle_bundle()

    print("\n" + "="*75)
    print("            PICKLE (.PKL) MODEL INFERENCE")
    print("="*75)
    print(f"Input Description:\n\"{text}\"\n")

    res = classify_text(text, bundle)
    print(f"[*] PREDICTED DISCIPLINE : {res['prediction'].upper()}")
    print(f"[*] CONFIDENCE SCORE     : {res['confidence']:.2f}%\n")

    prob_table = []
    for disc, prob in res["ranking"]:
        bar = "#" * int(prob / 4)
        prob_table.append([disc, f"{prob:.2f}%", bar])

    print(tabulate(prob_table, headers=["Discipline", "Probability", "Confidence Bar"], tablefmt="simple"))
    print("="*75 + "\n")

def run_interactive():
    bundle = load_pickle_bundle()

    print("\n" + "="*75)
    print("   INTERACTIVE .PKL CLASSIFIER (PRESS CTRL+C OR 'exit' TO QUIT)")
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

            res = classify_text(text, bundle)
            print("\n" + "-"*60)
            print(f"-> Predicted Category : {res['prediction']}")
            print(f"-> Confidence Score   : {res['confidence']:.2f}%")
            print("-" * 60)
            for disc, prob in res["ranking"]:
                bar = "#" * int(prob / 4)
                print(f"   {disc:<22} : {prob:>6.2f}% | {bar}")
            print("-" * 60)
        except KeyboardInterrupt:
            print("\nExiting interactive testing. Goodbye!")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Best Model saved in .pkl format.")
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
        run_suite()

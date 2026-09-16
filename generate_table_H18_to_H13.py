import os
import json
import numpy as np

MODELS = ["GAHT", "3D-CNN", "A-BLSTM", "DFFN", "M3DDCNN", "RSSAN", "SpeFormer", "SSFTT", "BiDA", "AgentBiDA", "PCADA", "TSTnet", "MDGTnet", "CLDA", "SCLUDA", "SSWADA", "CACL", "MLUDA"]
SEEDS = [678, 681, 774, 789]
NUM_CLASSES = 7

# Structure to hold metrics
results = {model: {'classes': {c: [] for c in range(1, NUM_CLASSES + 1)}, 'OA': [], 'AA': [], 'Kappa': []} for model in MODELS}

RESULTS_DIR = 'ablation_results_H18_H13'

if not os.path.exists(RESULTS_DIR):
    print(f"{RESULTS_DIR} directory not found.")
    exit(1)

# Load data
for model in MODELS:
    for seed in SEEDS:
        filepath = os.path.join(RESULTS_DIR, f"{model}_results_seed_{seed}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results[model]['OA'].append(data['OA'])
                    results[model]['AA'].append(data['AA'])
                    results[model]['Kappa'].append(data['Kappa'])
                    for c in range(1, NUM_CLASSES + 1):
                        results[model]['classes'][c].append(data['classes'][str(c)])
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

# Formatting function: mean \pm std
def format_metric(vals):
    if len(vals) == 0:
        return "N/A"
    return f"{np.mean(vals):.2f} \\pm {np.std(vals):.2f}"

# Generate LaTeX Table
print("\\begin{table*}[htbp]")
print("\\centering")
print("\\caption{Class-Specific and Overall Accuracy (\\%) Comparison, Houston18 $\\rightarrow$ Houston13}")
print("\\resizebox{\\textwidth}{!}{")
print("\\begin{tabular}{l" + "c" * len(MODELS) + "}")
print("\\toprule")
print("Class / Metric & " + " & ".join(MODELS) + " \\\\")
print("\\midrule")

for c in range(1, NUM_CLASSES + 1):
    row = f"{c} & "
    row += " & ".join([format_metric(results[m]['classes'][c]) for m in MODELS])
    print(row + " \\\\")

print("\\midrule")
print("OA (\\%) & " + " & ".join([format_metric(results[m]['OA']) for m in MODELS]) + " \\\\")
print("AA (\\%) & " + " & ".join([format_metric(results[m]['AA']) for m in MODELS]) + " \\\\")
print("$\\kappa \\times 100$ & " + " & ".join([format_metric(results[m]['Kappa']) for m in MODELS]) + " \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("}")
print("\\end{table*}")

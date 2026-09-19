import os
import json
import numpy as np
import pandas as pd

MODELS = ["BiDA", "AgentBiDA", "SelfAttentionAgentBiDA"]
SEEDS = [2100, 2101, 2102]
NUM_CLASSES = 7

# Structure to hold metrics
results = {model: {'classes': {c: [] for c in range(1, NUM_CLASSES + 1)}, 'OA': [], 'AA': [], 'Kappa': []} for model in MODELS}

RESULTS_DIR = 'ablation_results'

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
                pass

def format_metric(vals):
    if len(vals) == 0:
        return "N/A"
    return f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"

data = []
for c in range(1, NUM_CLASSES + 1):
    row = [f"Class {c}"] + [format_metric(results[m]['classes'][c]) for m in MODELS]
    data.append(row)

data.append(["OA (%)"] + [format_metric(results[m]['OA']) for m in MODELS])
data.append(["AA (%)"] + [format_metric(results[m]['AA']) for m in MODELS])
data.append(["Kappa x 100"] + [format_metric(results[m]['Kappa']) for m in MODELS])

df = pd.DataFrame(data, columns=["Metric"] + MODELS)
print("\n" + "="*80)
print("FINAL EVALUATION RESULTS (100 Epochs, 3 Seeds)")
print("="*80)
print(df.to_string(index=False))
print("="*80 + "\n")

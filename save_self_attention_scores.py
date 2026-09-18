import json
import os

seeds = [678, 681, 774, 789]
runs = [
    {
        "model": "SelfAttentionAgentBiDA",
        "seed": 678,
        "OA": 75.90833551998601,
        "AA": 69.8330390654182,
        "Kappa": 59.23096241477954,
        "Class_1": 83.53272343420126,
        "Class_2": 73.42937174869948,
        "Class_3": 66.73676831405538,
        "Class_4": 79.16666666666666,
        "Class_5": 93.67588932806325,
        "Class_6": 88.06591353096067,
        "Class_7": 4.223940435280642
    },
    {
        "model": "SelfAttentionAgentBiDA",
        "seed": 681,
        "OA": 75.14912971223652,
        "AA": 53.66347264363994,
        "Kappa": 52.250133055291705,
        "Class_1": 59.88740323715693,
        "Class_2": 64.0656262505002,
        "Class_3": 19.453207150368033,
        "Class_4": 50,
        "Class_5": 85.24973050664751,
        "Class_6": 94.65443528978736,
        "Class_7": 2.3339060710194732
    },
    {
        "model": "SelfAttentionAgentBiDA",
        "seed": 774,
        "OA": 74.50537916557334,
        "AA": 57.12191541772267,
        "Kappa": 49.85076340732096,
        "Class_1": 54.39831104855736,
        "Class_2": 48.75950380152061,
        "Class_3": 40.16824395373291,
        "Class_4": 75,
        "Class_5": 80.84800574919151,
        "Class_6": 94.17877063337015,
        "Class_7": 6.500572737686141
    },
    {
        "model": "SelfAttentionAgentBiDA",
        "seed": 789,
        "OA": 72.85926703402431,
        "AA": 59.581362087293414,
        "Kappa": 49.50787050698434,
        "Class_1": 90.4292751583392,
        "Class_2": 49.23969587835134,
        "Class_3": 62.9863301787592,
        "Class_4": 50,
        "Class_5": 73.33812432626662,
        "Class_6": 90.53200826750474,
        "Class_7": 0.5441008018327605
    }
]

os.makedirs('ablation_results', exist_ok=True)
for run in runs:
    filename = f"ablation_results/SelfAttentionAgentBiDA_results_seed_{run['seed']}.json"
    with open(filename, 'w') as f:
        json.dump(run, f, indent=4)
        print(f"Saved {filename}")

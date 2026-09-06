import subprocess
import re
import numpy as np
import argparse

def parse_output(output_str):
    # Extracts OA, AA, Kappa
    oa_match = re.search(r"OA:\s+([\d\.]+)", output_str)
    aa_match = re.search(r"AA:\s+([\d\.]+)", output_str)
    kappa_match = re.search(r"Kappa:\s+([\d\.]+)", output_str)
    
    if not (oa_match and aa_match and kappa_match):
        return None, None, None, None

    oa = float(oa_match.group(1))
    aa = float(aa_match.group(1))
    kappa = float(kappa_match.group(1))
    
    # Extracts classwise recall
    class_acc = []
    lines = output_str.split('\n')
    in_report = False
    for line in lines:
        if "Classification Report:" in line:
            in_report = True
            continue
        if in_report:
            if line.strip().startswith('accuracy') or line.strip() == '':
                if len(class_acc) >= 7:
                    break
            parts = line.strip().split()
            # If line starts with a number (0-6)
            if len(parts) >= 4 and parts[0].isdigit():
                class_acc.append(float(parts[2]) * 100) # Convert recall to percentage
                
    if len(class_acc) != 7:
        print(f"Warning: Expected 7 classes, found {len(class_acc)}")
                
    return oa, aa, kappa, class_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_name', type=str, default="Houston13")
    parser.add_argument('--target_name', type=str, default="Houston18")
    args = parser.parse_args()

    seeds = [2100, 2101, 2102, 2103]
    source = args.source_name
    target = args.target_name
    epochs = 200
    heads = 8
    num_agents = 4
    
    all_oa = []
    all_aa = []
    all_kappa = []
    all_class_acc = {i: [] for i in range(7)}
    
    print(f"Running GatedAgentBiDA for {source}->{target} over 4 seeds...")
    
    for seed in seeds:
        print(f"  Training Seed {seed}...")
        cmd = [
            "python3", "train_gated_agent_bida.py",
            "--source_name", source,
            "--target_name", target,
            "--epoch", str(epochs),
            "--seed", str(seed),
            "--num_agents", str(num_agents),
            "--num_heads", str(heads),
            "--gate_hidden_ratio", "0.25",
            "--gate_init_bias", "0.0"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        oa, aa, kappa, class_acc = parse_output(result.stdout)
        
        if oa is None:
            print(f"  Failed to parse output for seed {seed}. Check logs.")
            print(result.stdout)
            continue
            
        all_oa.append(oa)
        all_aa.append(aa)
        all_kappa.append(kappa)
        for i, acc in enumerate(class_acc):
            all_class_acc[i].append(acc)
            
    print("\n================ LaTeX Column for GatedAgentBiDA ================\n")
    
    for i in range(7):
        mean = np.mean(all_class_acc[i])
        std = np.std(all_class_acc[i], ddof=1) if len(all_class_acc[i]) > 1 else 0.0
        print(f"{mean:.2f} \pm {std:.2f}")
    
    print("\\hline")
    
    oa_mean = np.mean(all_oa)
    oa_std = np.std(all_oa, ddof=1) if len(all_oa) > 1 else 0.0
    print(f"{oa_mean:.2f} \pm {oa_std:.2f}")
    
    aa_mean = np.mean(all_aa)
    aa_std = np.std(all_aa, ddof=1) if len(all_aa) > 1 else 0.0
    print(f"{aa_mean:.2f} \pm {aa_std:.2f}")
    
    kappa_mean = np.mean(all_kappa) * 100
    kappa_std = np.std(all_kappa, ddof=1) * 100 if len(all_kappa) > 1 else 0.0
    print(f"{kappa_mean:.2f} \pm {kappa_std:.2f}")
    
    print("\n=================================================================\n")

if __name__ == "__main__":
    main()

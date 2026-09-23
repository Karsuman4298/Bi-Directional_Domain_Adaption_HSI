import os
import re
import numpy as np

def parse_logs(model, optimizer, seeds=[2100, 2101, 2102]):
    class_scores = {i: [] for i in range(7)}
    oa_scores = []
    aa_scores = []
    kappa_scores = []
    
    for seed in seeds:
        log_file = f"optimizer_test_18to13_{model}_{optimizer}_{seed}.log"
        if not os.path.exists(log_file):
            print(f"Warning: {log_file} not found!")
            continue
            
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Find OA, AA, Kappa
            oa_match = re.search(r"OA:\s*([0-9\.]+)", content)
            aa_match = re.search(r"AA:\s*([0-9\.]+)", content)
            kappa_match = re.search(r"Kappa:\s*([0-9\.]+)", content)
            
            if oa_match: oa_scores.append(float(oa_match.group(1)))
            if aa_match: aa_scores.append(float(aa_match.group(1)))
            if kappa_match: kappa_scores.append(float(kappa_match.group(1)))
            
            # Find class recall scores from classification report
            for class_idx in range(7):
                pattern = rf"^\s*{class_idx}\s+[0-9\.]+\s+([0-9\.]+)"
                match = re.search(pattern, content, re.MULTILINE)
                if match:
                    class_scores[class_idx].append(float(match.group(1)) * 100)
                    
    return class_scores, oa_scores, aa_scores, kappa_scores

def format_latex(name, class_scores, oa, aa, kappa):
    print(f"\n--- {name} ---")
    for i in range(7):
        scores = class_scores[i]
        if len(scores) > 0:
            mean = np.mean(scores)
            std = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
            print(f"Class {i+1} & {mean:.2f} \pm {std:.2f} \\\\")
        else:
            print(f"Class {i+1} & N/A \\\\")
            
    print("\\midrule")
    if oa:
        m_oa, s_oa = np.mean(oa), np.std(oa, ddof=1) if len(oa)>1 else 0
        print(f"OA (\%) & {m_oa:.2f} \pm {s_oa:.2f} \\\\")
    if aa:
        m_aa, s_aa = np.mean(aa), np.std(aa, ddof=1) if len(aa)>1 else 0
        print(f"AA (\%) & {m_aa:.2f} \pm {s_aa:.2f} \\\\")
    if kappa:
        # Check if Kappa is already a percentage or decimal
        m_k = np.mean(kappa)
        s_k = np.std(kappa, ddof=1) if len(kappa)>1 else 0
        if m_k < 1.0: 
            m_k *= 100
            s_k *= 100
        print(f"$\kappa \\times 100$ & {m_k:.2f} \pm {s_k:.2f} \\\\")

if __name__ == "__main__":
    c_b, o_b, a_b, k_b = parse_logs("BiDA", "SGD")
    format_latex("BiDA (SGD)", c_b, o_b, a_b, k_b)
    
    c_sa, o_sa, a_sa, k_sa = parse_logs("SelfAttentionAgentBiDA", "Adam")
    format_latex("Self-Attn Agent-BiDA (Adam)", c_sa, o_sa, a_sa, k_sa)

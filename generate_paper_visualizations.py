import argparse
import os
import random
import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as io

from utils.dataset import load_mat_hsi, sample_gt, HSIDataset
from utils.utils_HSI import seed_worker, count_sliding_window, sliding_window, grouper, metrics
from utils.scheduler import load_scheduler
from loss import make_loss
from train_pipeline import train, train_standard
from models.get_model import get_model

def run_experiment(opts):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{opts.device}")
    else:
        device = torch.device("cpu")
        
    print(f"\n========================================")
    print(f"Running Experiment for Source: {opts.source_name}, Target: {opts.target_name}")
    print(f"Models: {opts.models}")
    print(f"Seeds: {opts.seeds}")
    print(f"Epochs: {opts.epoch}")
    print(f"========================================\n")

    os.makedirs(opts.output_dir, exist_ok=True)
    
    # Storage for all models' metrics
    # structure: model_results[model_name] = {'oa': [], 'aa': [], 'kappa': [], 'class_accs': []}
    model_results = {}
    
    # Load dataset to get structure
    img_src_orig, gt_src_orig, labels_src = load_mat_hsi(opts.source_name, opts.dataset_dir, norm='normband')
    img_tar_orig, gt_tar_orig, labels_tar = load_mat_hsi(opts.target_name, opts.dataset_dir, norm='normband')
    
    num_classes = len(labels_tar)
    
    for model_name in opts.models:
        print(f"\n*******************************************************")
        print(f" Evaluating Model: {model_name}")
        print(f"*******************************************************")
        
        is_bida = model_name in ['BiDA', 'AgentBiDA', 'BiDA_Agent']
        
        model_results[model_name] = {'oa': [], 'aa': [], 'kappa': [], 'class_accs': []}
        
        for seed in opts.seeds:
            print(f"\n---> Starting run for {model_name} with seed {seed}")
            seed_worker(seed)
            
            # Reset generators
            g = torch.Generator()
            g.manual_seed(seed)
            
            # Prepare data with current seed
            train_gt_src, val_gt_src = sample_gt(gt_src_orig, opts.ratio, seed, mode='random')
            test_gt_tar, _ = sample_gt(gt_tar_orig, 1, seed, mode='random')
            
            img_src_con, train_gt_src_con = img_src_orig, train_gt_src
            val_gt_src_con = val_gt_src
            
            for i in range(opts.re_ratio - 1):
                img_src_con = np.concatenate((img_src_con, img_src_orig))
                train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
                val_gt_src_con = np.concatenate((val_gt_src_con, val_gt_src))

            r = opts.patch_size // 2
            img_src_pad = np.pad(img_src_con, ((r, r), (r, r), (0, 0)), mode='reflect')
            train_gt_src_pad = np.pad(train_gt_src_con, ((r, r), (r, r)), mode='reflect')
            val_gt_src_pad = np.pad(val_gt_src_con, ((r, r), (r, r)), mode='reflect')
            
            img_tar_pad = np.pad(img_tar_orig, ((r, r), (r, r), (0, 0)), mode='reflect')
            test_gt_tar_pad = np.pad(test_gt_tar, ((r, r), (r, r)), mode='reflect')

            train_set = HSIDataset(img_src_pad, train_gt_src_pad, patch_size=opts.patch_size, data_aug=True)
            val_set = HSIDataset(img_src_pad, val_gt_src_pad, patch_size=opts.patch_size, data_aug=False)
            
            train_loader = torch.utils.data.DataLoader(train_set, opts.bs, generator=g, drop_last=False, shuffle=True, num_workers=opts.num_workers)
            val_loader = torch.utils.data.DataLoader(val_set, opts.bs, generator=g, drop_last=False, shuffle=False, num_workers=opts.num_workers)

            test_dataset = HSIDataset(img_tar_pad, test_gt_tar_pad, patch_size=opts.patch_size, data_aug=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, opts.bs, generator=g, drop_last=False, shuffle=False, num_workers=opts.num_workers)

            if is_bida:
                test_dataset_noise = HSIDataset(img_tar_pad, test_gt_tar_pad, patch_size=opts.patch_size, data_aug=True)
                test_loader_noise = torch.utils.data.DataLoader(test_dataset_noise, opts.bs, generator=g, drop_last=False, shuffle=True, num_workers=opts.num_workers)
            else:
                test_loader_noise = None
                
            # Initialize Models
            model = get_model(model_name, opts.target_name, opts.patch_size, opts).to(device)
            
            if is_bida:
                model_ema = get_model(model_name, opts.target_name, opts.patch_size, opts, ema=True).to(device)
            else:
                model_ema = None
                
            optimizer, scheduler = load_scheduler(model_name, model, opts)
            
            if is_bida:
                criterion, _ = make_loss(opts, num_classes=num_classes, device=device)
            else:
                criterion = torch.nn.CrossEntropyLoss()
            
            # Override opts seed for this iteration
            opts.seed = seed
            
            checkpoint_dir = os.path.join(opts.output_dir, f'{model_name}_checkpoints_seed_{seed}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Train
            if is_bida:
                train(model, model_ema, optimizer, criterion, num_classes, 
                      train_loader, val_loader, test_loader_noise, test_loader, 
                      opts, checkpoint_dir, device, scheduler)
            else:
                train_standard(model, optimizer, criterion, num_classes,
                               train_loader, test_loader, opts, checkpoint_dir, device, scheduler)
                  
            # Find best checkpoint for this seed
            model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_ts_best') and f.endswith('.pth')]
            if not model_files:
                print(f"Warning: No checkpoint found for {model_name} seed {seed}!")
                continue
                
            best_model_file = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))[-1]
            checkpoint_path = os.path.join(checkpoint_dir, best_model_file)
            
            print(f"Loading best checkpoint for inference: {best_model_file}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            
            # Full Image Inference
            batch_size = 128
            window_size = (opts.patch_size, opts.patch_size)
            image_w, image_h = img_tar_orig.shape[:2]
            pad_size = opts.patch_size // 2

            probs = np.zeros(img_tar_pad.shape[:2] + (num_classes, ))

            iterations = count_sliding_window(img_tar_pad, step=1, window_size=window_size) // batch_size
            print(f'Running full-image inference for {model_name} classification map...')
            
            for batch in grouper(batch_size, sliding_window(img_tar_pad, step=1, window_size=window_size)):
                with torch.no_grad():
                    data = [b[0] for b in batch]
                    data = np.copy(data).transpose((0, 3, 1, 2))
                    data = torch.from_numpy(data).float().unsqueeze(1).to(device)
                    indices = [b[1:] for b in batch]
                    
                    if is_bida:
                        output = model(data, data) 
                        if isinstance(output, tuple):
                            output = output[1]
                    else:
                        output = model(data)
                        if isinstance(output, tuple):
                            output = output[0]
                        
                    output = output.cpu().numpy()

                    for (x, y, w, h), out in zip(indices, output):
                        probs[x + w // 2, y + h // 2] += out
                        
            probs = probs[pad_size:image_w + pad_size, pad_size:image_h + pad_size, :]
            pred_map = np.argmax(probs, axis=-1)

            pred_map_masked = np.copy(pred_map)
            pred_map_masked[gt_tar_orig == -1] = -1

            valid_mask = gt_tar_orig >= 0
            y_true = gt_tar_orig[valid_mask]
            y_pred = pred_map[valid_mask]

            results = metrics(y_pred, y_true, n_classes=num_classes)
            
            oa = results['Accuracy']
            aa = np.mean(results['TPR']) * 100
            kappa = results['Kappa']
            class_accs = results['TPR'] * 100
            
            model_results[model_name]['oa'].append(oa)
            model_results[model_name]['aa'].append(aa)
            model_results[model_name]['kappa'].append(kappa)
            model_results[model_name]['class_accs'].append(class_accs)
            model_results[model_name]['pred_map'] = pred_map_masked
            
            # Plot Confusion Matrix
            plt.figure(figsize=(10, 8))
            cm = results['Confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels_tar, yticklabels=labels_tar)
            plt.title(f'{model_name} Confusion Matrix - Seed {seed} (OA: {oa:.2f}%)')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig(os.path.join(opts.output_dir, f'{model_name}_confusion_matrix_seed_{seed}.png'), bbox_inches='tight', dpi=300)
            plt.close()

            # Plot Classification Map
            cmap = plt.get_cmap('jet', num_classes + 1)
            plt.figure(figsize=(16, 8))
            plt.subplot(1, 2, 1)
            plt.title(f'{model_name} Classification Map - Seed {seed}')
            plt.imshow(pred_map_masked, cmap=cmap, vmin=-1, vmax=num_classes-1)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title('Ground Truth Map')
            plt.imshow(gt_tar_orig, cmap=cmap, vmin=-1, vmax=num_classes-1)
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(opts.output_dir, f'{model_name}_classification_map_seed_{seed}.png'), bbox_inches='tight', dpi=300)
            plt.close()
        
    # Aggregate and Save Unified Report
    report_path = os.path.join(opts.output_dir, 'paper_results_table.txt')
    latex_path = os.path.join(opts.output_dir, 'paper_results_latex.tex')
    
    with open(report_path, 'w') as f, open(latex_path, 'w') as l:
        f.write("=========================================================\n")
        f.write(f" UNIFIED RESULTS REPORT FOR RESEARCH PAPER\n")
        f.write(f" Source: {opts.source_name}  |  Target: {opts.target_name}\n")
        f.write("=========================================================\n\n")
        
        # Determine the set of successful models
        successful_models = [m for m in opts.models if len(model_results[m]['oa']) > 0]
        
        if not successful_models:
            f.write("No successful runs completed for any model.\n")
            print("No successful runs completed. Skipping report generation.")
            return
            
        f.write(f"Tested Models: {', '.join(successful_models)}\n\n")
        
        # LaTeX Header
        l.write("\\begin{table*}[htbp]\n")
        l.write("\\centering\n")
        l.write("\\caption{Class-Specific and Overall Classification Accuracy (\\%) of Different Methods for the Target Scene " + opts.target_name + " Data}\n")
        l.write("\\label{tab:results_" + opts.target_name + "}\n")
        
        col_format = "c|" * (len(successful_models) + 1)
        l.write("\\begin{tabular}{|c|" + col_format + "}\n")
        l.write("\\hline\n")
        
        # Model Headers
        l.write("Class & " + " & ".join(successful_models) + " \\\\\n")
        l.write("\\hline\n")
        
        # 2. Class-specific Accuracy Comparison Table (Text and LaTeX)
        f.write("--- Class-Specific Accuracy (%) Comparison ---\n")
        class_header = f"{'Class':<5} | {'Name':<25} | " + " | ".join([f"{m:<20}" for m in successful_models])
        f.write(class_header + "\n")
        f.write("-" * len(class_header) + "\n")
        
        for i, label in enumerate(labels_tar):
            row_txt = f"{i+1:<5} | {label:<25} | "
            row_tex = f"{i+1} & "
            tex_vals = []
            
            for m in successful_models:
                class_accs_array = np.array(model_results[m]['class_accs'])
                mean_acc = np.mean(class_accs_array, axis=0)[i]
                std_acc = np.std(class_accs_array, axis=0)[i]
                row_txt += f"{mean_acc:.2f} ± {std_acc:.2f}".ljust(20) + " | "
                if len(opts.seeds) > 1:
                    tex_vals.append(f"{mean_acc:.2f}$\\pm${std_acc:.2f}")
                else:
                    tex_vals.append(f"{mean_acc:.2f}")
                    
            f.write(row_txt + "\n")
            l.write(row_tex + " & ".join(tex_vals) + " \\\\\n")
            
        f.write("\n")
        l.write("\\hline\n")
        
        # 1. Overall Metrics Comparison Table (Text and LaTeX)
        f.write("--- Overall Metrics Comparison ---\n")
        header = f"{'Metric':<15} | " + " | ".join([f"{m:<20}" for m in successful_models])
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        row_oa = f"{'OA (%)':<15} | "
        row_aa = f"{'AA (%)':<15} | "
        row_ka = f"{'Kappa (x100)':<15} | "
        
        tex_oa = "OA (\\%) & "
        tex_aa = "AA (\\%) & "
        tex_ka = "KC ($\\kappa$) & "
        
        vals_oa = []
        vals_aa = []
        vals_ka = []
        
        for m in successful_models:
            oa_m = np.mean(model_results[m]['oa']) * 100
            oa_s = np.std(model_results[m]['oa']) * 100
            aa_m = np.mean(model_results[m]['aa'])
            aa_s = np.std(model_results[m]['aa'])
            ka_m = np.mean(model_results[m]['kappa']) * 100
            ka_s = np.std(model_results[m]['kappa']) * 100
            
            row_oa += f"{oa_m:.2f} ± {oa_s:.2f}".ljust(20) + " | "
            row_aa += f"{aa_m:.2f} ± {aa_s:.2f}".ljust(20) + " | "
            row_ka += f"{ka_m:.2f} ± {ka_s:.2f}".ljust(20) + " | "
            
            if len(opts.seeds) > 1:
                vals_oa.append(f"\\textbf{{{oa_m:.2f}}}$\\pm${oa_s:.2f}" if m == 'AgentBiDA' else f"{oa_m:.2f}$\\pm${oa_s:.2f}")
                vals_aa.append(f"\\textbf{{{aa_m:.2f}}}$\\pm${aa_s:.2f}" if m == 'AgentBiDA' else f"{aa_m:.2f}$\\pm${aa_s:.2f}")
                vals_ka.append(f"\\textbf{{{ka_m:.2f}}}$\\pm${ka_s:.2f}" if m == 'AgentBiDA' else f"{ka_m:.2f}$\\pm${ka_s:.2f}")
            else:
                vals_oa.append(f"\\textbf{{{oa_m:.2f}}}" if m == 'AgentBiDA' else f"{oa_m:.2f}")
                vals_aa.append(f"\\textbf{{{aa_m:.2f}}}" if m == 'AgentBiDA' else f"{aa_m:.2f}")
                vals_ka.append(f"\\textbf{{{ka_m:.2f}}}" if m == 'AgentBiDA' else f"{ka_m:.2f}")
            
        f.write(row_oa + "\n")
        f.write(row_aa + "\n")
        f.write(row_ka + "\n\n")
        
        l.write(tex_oa + " & ".join(vals_oa) + " \\\\\n")
        l.write(tex_aa + " & ".join(vals_aa) + " \\\\\n")
        l.write(tex_ka + " & ".join(vals_ka) + " \\\\\n")
        l.write("\\hline\n")
        l.write("\\end{tabular}\n")
        l.write("\\end{table*}\n")

    # Generate the combined classification map figure
    if successful_models:
        print("Generating combined classification map figure...")
        num_plots = len(successful_models) + 1
        fig, axes = plt.subplots(1, num_plots, figsize=(2 * num_plots, 8))
        
        cmap = plt.get_cmap('jet', num_classes + 1)
        
        # 1. Plot Ground Truth
        axes[0].imshow(gt_tar_orig, cmap=cmap, vmin=-1, vmax=num_classes-1)
        axes[0].set_title("(a) Ground truth")
        axes[0].axis('off')
        
        # 2. Plot Models
        letters = "bcdefghijklmnopqrstuvwxyz"
        for idx, m in enumerate(successful_models):
            oa_m = np.mean(model_results[m]['oa']) * 100
            axes[idx + 1].imshow(model_results[m]['pred_map'], cmap=cmap, vmin=-1, vmax=num_classes-1)
            axes[idx + 1].set_title(f"({letters[idx]}) {m}\n({oa_m:.2f}%)")
            axes[idx + 1].axis('off')
            
        plt.tight_layout()
        combined_map_path = os.path.join(opts.output_dir, 'combined_classification_maps.png')
        plt.savefig(combined_map_path, bbox_inches='tight', dpi=300)
        plt.close()

    print("\n=========================================================")
    print(" Experiment Complete!")
    print(f" All outputs and report saved to: {opts.output_dir}")
    print("=========================================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Research Visualizations for BiDA and Baselines")
    parser.add_argument("--models", type=str, nargs='+', default=['AgentBiDA'], help='List of models to run')
    parser.add_argument('--source_name', type=str, default='Houston13', help='the name of the source dir')
    parser.add_argument('--target_name', type=str, default='Houston18', help='the name of the test dir')
    parser.add_argument("--dataset_dir", type=str, default='./Houston/')
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=13)
    parser.add_argument("--epoch", type=int, default=200)    
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--ratio", type=float, default=0.95)
    parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA')
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num_tokens", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--loss_type", type=str, default='softmax')
    parser.add_argument("--labelsmooth", type=str, default='off') 
    parser.add_argument("--lambda1", type=float, default=1e-1)
    parser.add_argument("--lambda2", type=float, default=1e+0)
    parser.add_argument("--log_interval", type=float, default=10)
    parser.add_argument('--re_ratio', type=int, default=1)
    parser.add_argument('--num_agents', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2100, 2101, 2102, 2103, 2104], help='List of random seeds to run')
    parser.add_argument('--output_dir', type=str, default='./paper_visualizations')

    opts = parser.parse_args()
    run_experiment(opts)

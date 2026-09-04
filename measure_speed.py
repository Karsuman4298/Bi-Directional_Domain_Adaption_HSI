import torch
import time
from models.BiDA import BiDAnet
from models.agent_bida import AgentBiDAnet

def measure_time(model, inputs, is_training=False, iterations=50, device='cpu'):
    model = model.to(device)
    if is_training:
        model.train()
    else:
        model.eval()

    # Warmup to avoid initialization overhead
    for _ in range(5):
        if is_training:
            out = model(inputs[0], inputs[1])
            loss = sum([o.sum() for o in out])
            loss.backward()
        else:
            with torch.no_grad():
                model(inputs[0], inputs[1], inference_target_only=True)

    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        if is_training:
            out = model(inputs[0], inputs[1])
            loss = sum([o.sum() for o in out])
            loss.backward()
        else:
            with torch.no_grad():
                model(inputs[0], inputs[1], inference_target_only=True)
                
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    return (end_time - start_time) / iterations

if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    batch_size = 128
    patch_size = 13
    n_bands = 48
    num_classes = 7
    dim = 64
    num_tokens = 4
    depth = 3
    heads = 8
    num_agents = 4
    
    print("\nInitializing models...")
    bida = BiDAnet(n_bands=n_bands, num_classes=num_classes, num_tokens=num_tokens, dim=dim, depth=depth, heads=heads).to(device)
    agent_bida = AgentBiDAnet(n_bands=n_bands, num_classes=num_classes, num_tokens=num_tokens, dim=dim, depth=depth, heads=heads, num_agents=num_agents).to(device)
    
    x_src = torch.randn(batch_size, 1, n_bands, patch_size, patch_size).to(device)
    x_tar = torch.randn(batch_size, 1, n_bands, patch_size, patch_size).to(device)
    
    iterations = 50
    
    print("\n--- Measuring Inference Speed (Target Only) ---")
    bida_inf_time = measure_time(bida, (x_src, x_tar), is_training=False, iterations=iterations, device=device)
    agent_inf_time = measure_time(agent_bida, (x_src, x_tar), is_training=False, iterations=iterations, device=device)
    
    print(f"Original BiDA Inference Time: {bida_inf_time*1000:.2f} ms/batch")
    print(f"AgentBiDA Inference Time:     {agent_inf_time*1000:.2f} ms/batch")
    
    if agent_inf_time < bida_inf_time:
        speedup = ((bida_inf_time - agent_inf_time) / bida_inf_time) * 100
        print(f"AgentBiDA is {speedup:.2f}% FASTER in inference.")
    else:
        slowdown = ((agent_inf_time - bida_inf_time) / bida_inf_time) * 100
        print(f"AgentBiDA is {slowdown:.2f}% SLOWER in inference.")
        
    print("\n--- Measuring Training Speed (Forward + Backward) ---")
    bida_train_time = measure_time(bida, (x_src, x_tar), is_training=True, iterations=iterations, device=device)
    agent_train_time = measure_time(agent_bida, (x_src, x_tar), is_training=True, iterations=iterations, device=device)
    
    print(f"Original BiDA Training Time: {bida_train_time*1000:.2f} ms/batch")
    print(f"AgentBiDA Training Time:     {agent_train_time*1000:.2f} ms/batch")
    
    if agent_train_time < bida_train_time:
        speedup = ((bida_train_time - agent_train_time) / bida_train_time) * 100
        print(f"AgentBiDA is {speedup:.2f}% FASTER in training.")
    else:
        slowdown = ((agent_train_time - bida_train_time) / bida_train_time) * 100
        print(f"AgentBiDA is {slowdown:.2f}% SLOWER in training.")

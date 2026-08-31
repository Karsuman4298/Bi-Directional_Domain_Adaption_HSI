# AgentBiDA: Query-Derived Agent Attention for Bi-directional Domain Adaptation

This document explains the architecture and mathematical foundation of the `AgentBiDA` implementation, which replaces the standard Transformer attention mechanism in the original BiDA with **Agent Attention**.

## Architecture Overview

AgentBiDA preserves the Bi-directional Domain Adaptation workflow while modifying the core feature representations using a bottleneck attention mechanism.

```
                    HSI PATCH
                       |
                       v
                    Conv3D
                       |
                       v
                    Conv2D
                       |
                       v
                Feature Embedding
                       |
                       v
                 Semantic Tokens (4)
                       |
                       +------ Positional Embedding
                       |
                       +------ CLS Token
                       |
                       v
             Agent Transformer Block (3 layers)
                       |
                       +-----------------------+
                       |                       |
                       v                       |
                  LayerNorm                   |
                       |                       |
                       v                       |
                       Q                       |
                    /     \                    |
                   /       \                   |
                  v         v                  |
              Pooling      K,V                 |
                  |         |                  |
                  v         |                  |
                  A         |                  |
                  |         |                  |
                  +----+----+                  |
                       |                       |
                       v                       |
                  A K^T                        |
                       |                       |
                    Softmax                    |
                       |                       |
                       v                       |
                  Agent Features               |
                       |                       |
                  Q A^T                        |
                       |                       |
                    Softmax                    |
                       |                       |
                       v                       |
                    Output                    |
                       |                       |
                  Projection                   |
                       |                       |
                       v                       |
                    Residual <----------------+
                       |
                       v
                      MLP
                       |
                       v
                    Residual
                       |
                       v
                 Classification
                  / Domain Heads
                       |
                       v
              BiDA-compatible outputs
```

*Note: Since the sequence length (N=5) is very small in this specific codebase (due to the preceding semantic token compression), the Agent Attention mechanism performs 1D sequence interpolation to generate `num_agents` tokens, meaning `A` is dynamically resampled from `Q`.*

## Tensor Flow and Dimensions

### Input
- `X` ∈ ℝ^[B, N, D]
- In our specific implementation: `X` has shape `[B, 5, 64]`.

### Projection
Standard linear projections with dimension `D` and `H` heads, where `Dh = D/H`.
- `Q = XW_Q`
- `K = XW_K`
- `V = XW_V`
- Tensor Shape: `[B, H, N, Dh]` (e.g., `[B, 8, 5, 8]`)

### Agent Generation
Agents are generated dynamically by pooling the `Q` representations over the sequence length.
- `A = Pool(Q)`
- `A` ∈ ℝ^[B, H, n, Dh]
- In our codebase, with `num_agents = 4`, `A` has shape `[B, 8, 4, 8]`.

### Stage 1: Agent Aggregation (Token → Agent)
Agents aggregate information globally from `K` and `V`.
- `S_A = Softmax(A @ K^T / sqrt(Dh))`
- `S_A` ∈ ℝ^[B, H, n, N] (e.g., `[B, 8, 4, 5]`)
- `VA = S_A @ V`
- `VA` ∈ ℝ^[B, H, n, Dh] (e.g., `[B, 8, 4, 8]`)

### Stage 2: Agent Broadcast (Agent → Token)
The original tokens retrieve information from the summarized agent features `VA`.
- `S_Q = Softmax(Q @ A^T / sqrt(Dh))`
- `S_Q` ∈ ℝ^[B, H, N, n] (e.g., `[B, 8, 5, 4]`)
- `Y = S_Q @ VA`
- `Y` ∈ ℝ^[B, H, N, Dh] (e.g., `[B, 8, 5, 8]`)
- Concatenated output back to `[B, N, D]`.

## Computational Complexity

- **Standard Attention**: Requires computing `N x N` attention matrices, yielding complexity **O(N²D)**.
- **Agent Attention**: Computes `n x N` (Aggregation) and `N x n` (Broadcast) matrices, yielding complexity **O(NnD)**.
  
Because `n` (num_agents) is a hyperparameter and usually `n << N`, this attention mechanism is linear with respect to sequence length.

## Ablation Study Support

The `train_agent_bida.py` script has been decoupled from the original `main.py` specifically for running cleanly isolated experiments.

### Running Baseline BiDA
```bash
python main.py --model BiDA
```

### Running AgentBiDA Configurations
Run AgentBiDA with standard configuration (4 agents, 8 heads):
```bash
python train_agent_bida.py --num_agents 4 --num_heads 8
```

Increase the number of agents to see scaling properties:
```bash
python train_agent_bida.py --num_agents 8
python train_agent_bida.py --num_agents 16
python train_agent_bida.py --num_agents 32
```

Use `--debug_shapes` to print the exact tensor flow without initiating training:
```bash
python train_agent_bida.py --debug_shapes --num_agents 16
```

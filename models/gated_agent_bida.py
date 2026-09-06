"""
Confidence-Gated Dual-Attention AgentBiDA (GatedAgentBiDA)
============================================================

Motivation (grounded in your own ablation results):
  - AgentBiDA (agent-bottleneck on source-self, target-self, AND fusion)
    wins OA / kappa / stability, but is weak & high-variance on small,
    spectrally-distinct classes (e.g. Water: 70.45 +/- 19.68).
  - SelfAttnAgentBiDA (full self-attention on source-self, target-self;
    agent-bottleneck only on fusion) recovers that class (Water: 77.27
    +/- 7.87) but loses a bit of OA / kappa / AA overall.

Neither variant is uniformly better -> the fix is not "pick one" but
"let the network learn, per-sample, how much to trust the compressed
agent summary vs. the exact full self-attention representation" for
the source-self and target-self branches. The fusion (cross-domain)
branch is kept fully agent-based, since that's the anchor for
AgentBiDA's OA / stability advantage and isn't implicated in the
water/road weakness.

    x_out = g * x_agent + (1 - g) * x_full,   g = sigmoid(MLP(mean_pool(x)))

g -> 1 : trust the agent-compressed (efficient, majority-class-friendly)
         representation
g -> 0 : trust the full, exact self-attention (better for small /
         spectrally-distinct classes) representation

This is a drop-in replacement for `AgentAttention` / `AgentTransformerBlock`
/ `AgentBiDAnet` from your existing `models/self_attention_agent_bida.py`
and `models/agent_bida.py`. Variable names / shapes match those files so
you can diff against them directly.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


# ----------------------------------------------------------------------
# DropPath / Mlp — unchanged from your existing files
# ----------------------------------------------------------------------
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ----------------------------------------------------------------------
# Core novelty: Confidence-Gated Dual-Attention
# ----------------------------------------------------------------------
class GatedAgentAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_agents=4, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., gate_hidden_ratio=0.25, gate_init_bias=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_agents = num_agents
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # --- Confidence gate ---
        # Pools the branch's own token sequence to a per-sample summary,
        # then predicts a scalar in (0,1): how much to trust the
        # agent-compressed representation vs. the full one.
        gate_hidden = max(int(dim * gate_hidden_ratio), 8)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        # Zero-init the final layer so training starts at an unbiased
        # 50/50 mix (sigmoid(gate_init_bias)) rather than favoring one
        # attention style before any signal has been learned.
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, gate_init_bias)

    # ---- shared agent-pooling, identical strategy to your existing code ----
    def _pool_1d(self, x, num_agents):
        # x: [B*H, D_h, N]
        cls_token = x[:, :, 0:1]
        patch_tokens = x[:, :, 1:]
        pool_size = num_agents - 1
        if pool_size <= 0:
            return cls_token
        if x.device.type == 'mps':
            pooled_patches = F.interpolate(patch_tokens, size=pool_size, mode='linear', align_corners=False)
        else:
            pooled_patches = F.adaptive_avg_pool1d(patch_tokens, pool_size)
        return torch.cat((cls_token, pooled_patches), dim=2)

    def _gate(self, x):
        # x: [B, N, C] -> [B, 1, 1] gate value, broadcastable over tokens/heads
        pooled = x.mean(dim=1)                      # [B, C]
        g = torch.sigmoid(self.gate_mlp(pooled))     # [B, 1]
        return g.unsqueeze(1)                        # [B, 1, 1]

    def _full_self_attention(self, q, k, v, B, N, C):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return (attn @ v).transpose(1, 2).reshape(B, N, C)

    def _agent_self_attention(self, q, k, v, B, N, C):
        q_reshaped = q.reshape(B * self.num_heads, C // self.num_heads, N)
        agent_q = self._pool_1d(q_reshaped, self.num_agents).reshape(
            B, self.num_heads, C // self.num_heads, self.num_agents).transpose(2, 3)  # [B,H,A,D_h]

        # Stage 1: token -> agent aggregation
        attn_agent = (agent_q @ k.transpose(-2, -1)) * self.scale
        attn_agent = attn_agent.softmax(dim=-1)
        attn_agent = self.attn_drop(attn_agent)
        VA = attn_agent @ v                                   # [B,H,A,D_h]

        # Stage 2: agent -> token broadcast
        attn_feature = (q @ agent_q.transpose(-2, -1)) * self.scale
        attn_feature = attn_feature.softmax(dim=-1)
        attn_feature = self.attn_drop(attn_feature)
        return (attn_feature @ VA).transpose(1, 2).reshape(B, N, C)

    def forward(self, x, x2, inference_target_only=False, debug_shapes=False, return_gates=False):
        B, N, C = x2.shape

        # -------- inference-time target-only fast path --------
        if inference_target_only:
            qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

            x2_full = self._full_self_attention(q2, k2, v2, B, N, C)
            x2_agent = self._agent_self_attention(q2, k2, v2, B, N, C)
            g_tgt = self._gate(x2)
            x2_out = g_tgt * x2_agent + (1 - g_tgt) * x2_full

            x2 = self.proj_drop(self.proj(x2_out))
            x, x3, x4 = None, None, None

            if debug_shapes:
                print(f"[GatedAgentAttention] inference gate (tgt) mean={g_tgt.mean().item():.3f}")

            if return_gates:
                return x, x2, x3, x4, {"g_src": None, "g_tgt": g_tgt}
            return x, x2, x3, x4

        # -------- full training-time path --------
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

        # ---- Source branch: full vs. agent, confidence-gated ----
        x_full = self._full_self_attention(q, k, v, B, N, C)
        x_agent = self._agent_self_attention(q, k, v, B, N, C)
        g_src = self._gate(x)
        x_out = g_src * x_agent + (1 - g_src) * x_full

        # ---- Target branch: full vs. agent, confidence-gated ----
        x2_full = self._full_self_attention(q2, k2, v2, B, N, C)
        x2_agent = self._agent_self_attention(q2, k2, v2, B, N, C)
        g_tgt = self._gate(x2)
        x2_out = g_tgt * x2_agent + (1 - g_tgt) * x2_full

        # ---- Fusion branch: kept FULLY agent-based (unchanged from AgentBiDA) ----
        # This is deliberately NOT gated: it's the anchor for AgentBiDA's
        # OA / stability advantage, and isn't implicated in the
        # water/road weakness this design targets.
        q_st = torch.cat((q, q2), dim=0)
        k_st = torch.cat((k2, k), dim=0)
        v_st = torch.cat((v2, v), dim=0)

        q_st_reshaped = q_st.reshape(2 * B * self.num_heads, C // self.num_heads, N)
        agent_q_st = self._pool_1d(q_st_reshaped, self.num_agents).reshape(
            2 * B, self.num_heads, C // self.num_heads, self.num_agents).transpose(2, 3)

        attn_agent_st = (agent_q_st @ k_st.transpose(-2, -1)) * self.scale
        attn_agent_st = attn_agent_st.softmax(dim=-1)
        attn_agent_st = self.attn_drop(attn_agent_st)
        VA_st = attn_agent_st @ v_st

        attn_feature_st = (q_st @ agent_q_st.transpose(-2, -1)) * self.scale
        attn_feature_st = attn_feature_st.softmax(dim=-1)
        attn_feature_st = self.attn_drop(attn_feature_st)
        x_st_out = (attn_feature_st @ VA_st).transpose(1, 2).reshape(2 * B, N, C)

        # ---- Projections ----
        x = self.proj_drop(self.proj(x_out))
        x2 = self.proj_drop(self.proj(x2_out))
        x_st = self.proj_drop(self.proj(x_st_out))
        x3, x4 = torch.split(x_st, B, dim=0)

        if debug_shapes:
            print(f"[GatedAgentAttention] src gate mean={g_src.mean().item():.3f}, "
                  f"tgt gate mean={g_tgt.mean().item():.3f}")

        if return_gates:
            return x, x2, x3, x4, {"g_src": g_src, "g_tgt": g_tgt}
        return x, x2, x3, x4


# ----------------------------------------------------------------------
# Transformer block wrapping GatedAgentAttention — mirrors AgentTransformerBlock
# ----------------------------------------------------------------------
class GatedAgentTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_agents=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 gate_hidden_ratio=0.25, gate_init_bias=0.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GatedAgentAttention(
            dim, num_heads=num_heads, num_agents=num_agents, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, gate_hidden_ratio=gate_hidden_ratio,
            gate_init_bias=gate_init_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x2, x1_x2_fusion, inference_target_only=False, debug_shapes=False, return_gates=False):
        if inference_target_only:
            attn_out = self.attn(None, self.norm1(x2), inference_target_only=True,
                                  debug_shapes=debug_shapes, return_gates=return_gates)
            if return_gates:
                _, xa_attn2, _, _, gates = attn_out
            else:
                _, xa_attn2, _, _ = attn_out
                gates = None
            xb = x2 + self.drop_path(xa_attn2)
            xb = xb + self.drop_path(self.mlp(self.norm2(xb)))
            xa, xab, xba = None, None, None
        else:
            attn_out = self.attn(self.norm1(x), self.norm1(x2), inference_target_only=False,
                                  debug_shapes=debug_shapes, return_gates=return_gates)
            if return_gates:
                xa_attn, xa_attn2, xa_attn3, xa_attn4, gates = attn_out
            else:
                xa_attn, xa_attn2, xa_attn3, xa_attn4 = attn_out
                gates = None

            xa = x + self.drop_path(xa_attn)
            xa = xa + self.drop_path(self.mlp(self.norm2(xa)))

            xb = x2 + self.drop_path(xa_attn2)
            xb = xb + self.drop_path(self.mlp(self.norm2(xb)))

            xab = x1_x2_fusion + self.drop_path(xa_attn3)
            xab = xab + self.drop_path(self.mlp(self.norm2(xab)))

            xba = x + self.drop_path(xa_attn4)
            xba = xba + self.drop_path(self.mlp(self.norm2(xba)))

        if return_gates:
            return xa, xb, xab, xba, gates
        return xa, xb, xab, xba


# ----------------------------------------------------------------------
# Full network — mirrors AgentBiDAnet, swaps in GatedAgentTransformerBlock
# ----------------------------------------------------------------------
class GatedAgentBiDAnet(nn.Module):
    def __init__(self, n_bands=30, in_channels=1, num_classes=16, num_tokens=4, dim=64, depth=1,
                 heads=8, num_agents=4, mlp_dim=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.,
                 gate_hidden_ratio=0.25, gate_init_bias=0.0, gate_reg_weight=0.0):
        super(GatedAgentBiDAnet, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.token_len = num_tokens
        self.gate_reg_weight = gate_reg_weight  # optional: penalize degenerate (all-0/all-1) gates

        self.conv_a = nn.Conv2d(dim, self.token_len, kernel_size=1, padding=0, bias=False)

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8 * n_bands, out_channels=dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            GatedAgentTransformerBlock(
                dim=dim, num_heads=heads, num_agents=num_agents, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=nn.LayerNorm,
                gate_hidden_ratio=gate_hidden_ratio, gate_init_bias=gate_init_bias)
            for i in range(depth)])
        self.norm = nn.LayerNorm(dim)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

        print("\n--- GatedAgentBiDA Architecture ---")
        print(f"num_agents: {num_agents}")
        print(f"num_heads: {heads}")
        print(f"embed_dim: {dim}")
        print(f"num_blocks: {depth}")
        print(f"gate_hidden_ratio: {gate_hidden_ratio}, gate_init_bias: {gate_init_bias}")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}\n")

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _tokenize(self, x):
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        T = self._forward_semantic_tokens(x)
        return T

    def forward(self, x, x_tar, inference_target_only=False, return_feat_prob=False,
                debug_shapes=False, return_gates=False):
        T = self._tokenize(x) if x is not None else None
        T_tar = self._tokenize(x_tar)

        cls_tokens = self.cls_token.expand(x_tar.shape[0], -1, -1)
        x_tar = torch.cat((cls_tokens, T_tar), dim=1)
        x_tar += self.pos_embedding
        x_tar = self.dropout(x_tar)

        if x is not None:
            x = torch.cat((cls_tokens, T), dim=1)
            x += self.pos_embedding
            x = self.dropout(x)

        inference_target_only = not self.training if not debug_shapes else inference_target_only
        x_fusion = x_tar
        all_gates = [] if return_gates else None

        for i, blk in enumerate(self.blocks):
            blk_out = blk(x, x_tar, x_fusion, inference_target_only=inference_target_only,
                           debug_shapes=debug_shapes, return_gates=return_gates)
            if return_gates:
                x, x_tar, x_fusion, x_fusion_src, gates = blk_out
                all_gates.append(gates)
            else:
                x, x_tar, x_fusion, x_fusion_src = blk_out

        # ---- optional gate regularization loss (encourages decisive, non-collapsed gates) ----
        gate_reg_loss = None
        if return_gates and self.gate_reg_weight > 0.0:
            reg_terms = []
            for gates in all_gates:
                for key in ("g_src", "g_tgt"):
                    g = gates.get(key)
                    if g is not None:
                        # penalize gates sitting exactly at 0.5 (uninformative / not using
                        # either attention style meaningfully) — pushes the gate to commit
                        reg_terms.append(-((g - 0.5) ** 2).mean())
            if reg_terms:
                gate_reg_loss = self.gate_reg_weight * torch.stack(reg_terms).mean()

        if inference_target_only:
            x_tar = self.norm(x_tar)
            out_x_tar = self.nn1(self.to_cls_token(x_tar[:, 0]))
            if return_feat_prob:
                out = (None, out_x_tar, None, x_tar[:, 0])
            else:
                out = (None, out_x_tar, None)
        else:
            x = self.norm(x)
            x_tar = self.norm(x_tar)
            x_fusion = self.norm(x_fusion)
            x_fusion_src = self.norm(x_fusion_src)
            out_x = self.nn1(self.to_cls_token(x[:, 0]))
            out_x_tar = self.nn1(self.to_cls_token(x_tar[:, 0]))
            out_x_fusion = self.nn1(self.to_cls_token(x_fusion[:, 0]))
            out_fusion_src = self.nn1(self.to_cls_token(x_fusion_src[:, 0]))
            out = (out_x, out_x_tar, out_x_fusion, out_fusion_src)

        if return_gates:
            return (*out, all_gates, gate_reg_loss)
        return out


# ----------------------------------------------------------------------
# Factory function — matches your existing AgentBiDA(dataset, opts) signature
# ----------------------------------------------------------------------
def GatedAgentBiDA(dataset, opts):
    model = None
    gate_hidden_ratio = getattr(opts, "gate_hidden_ratio", 0.25)
    gate_init_bias = getattr(opts, "gate_init_bias", 0.0)
    gate_reg_weight = getattr(opts, "gate_reg_weight", 0.0)

    if 'MJG' in dataset.split('_'):
        model = GatedAgentBiDAnet(
            n_bands=64, num_classes=5, num_tokens=opts.num_tokens, dim=opts.dim, depth=opts.depth,
            heads=opts.num_heads, num_agents=opts.num_agents,
            gate_hidden_ratio=gate_hidden_ratio, gate_init_bias=gate_init_bias,
            gate_reg_weight=gate_reg_weight)
    elif dataset == 'Houston18' or dataset == 'Houston13':
        model = GatedAgentBiDAnet(
            n_bands=48, num_classes=7, num_tokens=opts.num_tokens, dim=opts.dim, depth=opts.depth,
            heads=opts.num_heads, num_agents=opts.num_agents,
            gate_hidden_ratio=gate_hidden_ratio, gate_init_bias=gate_init_bias,
            gate_reg_weight=gate_reg_weight)
    elif dataset == 'Dioni' or dataset == 'Loukia':
        model = GatedAgentBiDAnet(
            n_bands=176, num_classes=12, num_tokens=opts.num_tokens, dim=opts.dim, depth=opts.depth,
            heads=opts.num_heads, num_agents=opts.num_agents,
            gate_hidden_ratio=gate_hidden_ratio, gate_init_bias=gate_init_bias,
            gate_reg_weight=gate_reg_weight)
    return model

import torch
from einops import rearrange
from torch import nn

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


class AgentAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_agents=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.num_agents = num_agents
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # We use AdaptiveAvgPool1d to dynamically pool the sequence of tokens into agent tokens
        # However, due to MPS limitations (input size must be divisible by output size), we wrap it
        self.num_agents = num_agents

    def _pool_1d(self, x, num_agents):
        # x is [B*H, D_h, N]
        if x.device.type == 'mps':
            # AdaptiveAvgPool1d is buggy on MPS for non-divisible sizes, so we fallback to CPU or interpolate
            # Using interpolate with nearest or linear
            import torch.nn.functional as F
            # Interpolate expects float type, we can use it to resize the sequence dimension
            return F.interpolate(x, size=num_agents, mode='linear', align_corners=False)
        else:
            import torch.nn.functional as F
            return F.adaptive_avg_pool1d(x, num_agents)

    def forward(self, x, x2, inference_target_only=False, debug_shapes=False):
        B, N, C = x2.shape
        if inference_target_only:
            qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

            # Generate agents from Q2
            q2_reshaped = q2.reshape(B * self.num_heads, C // self.num_heads, N)
            agent_q2 = self._pool_1d(q2_reshaped, self.num_agents).reshape(B, self.num_heads, C // self.num_heads, self.num_agents).transpose(2, 3)

            if debug_shapes:
                print(f"\n--- Agent Attention Debug Shapes ---")
                print(f"X              : {list(x2.shape)}")
                print(f"Q, K, V        : {list(q2.shape)}")
                print(f"A              : {list(agent_q2.shape)}")

            # Stage 1: Agent Aggregation (Tokens -> Agents)
            attn_agent2 = (agent_q2 @ k2.transpose(-2, -1)) * self.scale
            if debug_shapes:
                print(f"A K^T          : {list(attn_agent2.shape)}")
                
            attn_agent2 = attn_agent2.softmax(dim=-1)
            attn_agent2 = self.attn_drop(attn_agent2)
            VA2 = attn_agent2 @ v2
            if debug_shapes:
                print(f"VA             : {list(VA2.shape)}")

            # Stage 2: Agent Broadcast (Agents -> Tokens)
            attn_feature2 = (q2 @ agent_q2.transpose(-2, -1)) * self.scale
            if debug_shapes:
                print(f"Q A^T          : {list(attn_feature2.shape)}")
                
            attn_feature2 = attn_feature2.softmax(dim=-1)
            attn_feature2 = self.attn_drop(attn_feature2)
            x2_out = (attn_feature2 @ VA2).transpose(1, 2).reshape(B, N, C)
            if debug_shapes:
                print(f"Y              : {list(x2_out.shape)}")

            x2 = self.proj(x2_out)
            x2 = self.proj_drop(x2)
            if debug_shapes:
                print(f"Output         : {list(x2.shape)}")
                
            x, x3, x4 = None, None, None
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

            q_st = torch.cat((q, q2), dim=0)
            k_st = torch.cat((k2, k), dim=0)
            v_st = torch.cat((v2, v), dim=0)

            # Generate agents from queries
            q_reshaped = q.reshape(B * self.num_heads, C // self.num_heads, N)
            agent_q = self._pool_1d(q_reshaped, self.num_agents).reshape(B, self.num_heads, C // self.num_heads, self.num_agents).transpose(2, 3)

            q2_reshaped = q2.reshape(B * self.num_heads, C // self.num_heads, N)
            agent_q2 = self._pool_1d(q2_reshaped, self.num_agents).reshape(B, self.num_heads, C // self.num_heads, self.num_agents).transpose(2, 3)

            q_st_reshaped = q_st.reshape(2 * B * self.num_heads, C // self.num_heads, N)
            agent_q_st = self._pool_1d(q_st_reshaped, self.num_agents).reshape(2 * B, self.num_heads, C // self.num_heads, self.num_agents).transpose(2, 3)

            # Stage 1: Aggregation
            attn_agent = (agent_q @ k.transpose(-2, -1)) * self.scale
            attn_agent = attn_agent.softmax(dim=-1)
            attn_agent = self.attn_drop(attn_agent)
            VA = attn_agent @ v

            attn_agent2 = (agent_q2 @ k2.transpose(-2, -1)) * self.scale
            attn_agent2 = attn_agent2.softmax(dim=-1)
            attn_agent2 = self.attn_drop(attn_agent2)
            VA2 = attn_agent2 @ v2

            attn_agent_st = (agent_q_st @ k_st.transpose(-2, -1)) * self.scale
            attn_agent_st = attn_agent_st.softmax(dim=-1)
            attn_agent_st = self.attn_drop(attn_agent_st)
            VA_st = attn_agent_st @ v_st

            # Stage 2: Broadcast
            attn_feature = (q @ agent_q.transpose(-2, -1)) * self.scale
            attn_feature = attn_feature.softmax(dim=-1)
            attn_feature = self.attn_drop(attn_feature)
            x_out = (attn_feature @ VA).transpose(1, 2).reshape(B, N, C)

            attn_feature2 = (q2 @ agent_q2.transpose(-2, -1)) * self.scale
            attn_feature2 = attn_feature2.softmax(dim=-1)
            attn_feature2 = self.attn_drop(attn_feature2)
            x2_out = (attn_feature2 @ VA2).transpose(1, 2).reshape(B, N, C)

            attn_feature_st = (q_st @ agent_q_st.transpose(-2, -1)) * self.scale
            attn_feature_st = attn_feature_st.softmax(dim=-1)
            attn_feature_st = self.attn_drop(attn_feature_st)
            x_st_out = (attn_feature_st @ VA_st).transpose(1, 2).reshape(2*B, N, C)

            x = self.proj(x_out)
            x = self.proj_drop(x)

            x2 = self.proj(x2_out)
            x2 = self.proj_drop(x2)
            
            x_st = self.proj(x_st_out)
            x_st = self.proj_drop(x_st)
            x3, x4 = torch.split(x_st, B, dim=0)

        return x, x2, x3, x4

    
class AgentTransformerBlock(nn.Module):
    
    def __init__(self, dim, num_heads, num_agents=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AgentAttention(
            dim, num_heads=num_heads, num_agents=num_agents, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x, x2, x1_x2_fusion, inference_target_only=False, debug_shapes=False):
        if inference_target_only:
            _, xa_attn2, _, _ = self.attn(None, self.norm1(x2), inference_target_only=inference_target_only, debug_shapes=debug_shapes)
            xb = x2 + self.drop_path(xa_attn2)
            xb = xb + self.drop_path(self.mlp(self.norm2(xb)))
            xa, xab, xba = None, None, None
        else:
            xa_attn, xa_attn2, xa_attn3, xa_attn4 = self.attn(self.norm1(x), self.norm1(x2), inference_target_only=inference_target_only, debug_shapes=debug_shapes)
            xa = x + self.drop_path(xa_attn)
            xa = xa + self.drop_path(self.mlp(self.norm2(xa)))

            xb = x2 + self.drop_path(xa_attn2)
            xb = xb + self.drop_path(self.mlp(self.norm2(xb)))

            xab = x1_x2_fusion + self.drop_path(xa_attn3)
            xab = xab + self.drop_path(self.mlp(self.norm2(xab)))

            xba = x + self.drop_path(xa_attn4)
            xba = xba + self.drop_path(self.mlp(self.norm2(xba)))
            
        return xa, xb, xab, xba

class AgentBiDAnet(nn.Module):
    def __init__(self, n_bands=30, in_channels=1, num_classes=16, num_tokens=4, dim=64, depth=1, heads=8, num_agents=4, mlp_dim=8,  
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.):
        super(AgentBiDAnet, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.token_len = num_tokens
        self.conv_a = nn.Conv2d(dim, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8*n_bands, out_channels=dim,
                      kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            AgentTransformerBlock(
                dim=dim, num_heads=heads, num_agents=num_agents, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        
        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

        # Print parameters
        print("\n--- AgentBiDA Architecture ---")
        print(f"num_agents: {num_agents}")
        print(f"num_heads: {heads}")
        print(f"embed_dim: {dim}")
        print(f"num_blocks: {depth}")
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
    
    def forward(self, x, x_tar, inference_target_only=False, return_feat_prob=False, debug_shapes=False):
        if debug_shapes and inference_target_only:
            print(f"Input                 : {list(x_tar.shape)}")
            
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

        if debug_shapes and inference_target_only:
            print(f"After Semantic Tokens : {list(T_tar[:, 1:].shape)}")
            print(f"Transformer Input     : {list(x_tar.shape)}")
            
        inference_target_only = not self.training if not debug_shapes else inference_target_only
        x_fusion = x_tar
        for i, blk in enumerate(self.blocks):
            x, x_tar, x_fusion, x_fusion_src = blk(
                x, x_tar, x_fusion, inference_target_only=inference_target_only, debug_shapes=debug_shapes)
            
        if inference_target_only:
            x_tar = self.norm(x_tar)
            out_x_tar = self.nn1(self.to_cls_token(x_tar[:, 0])) 
            if return_feat_prob:
                return None, out_x_tar, None, x_tar[:, 0]
            else:
                return None, out_x_tar, None
        else:
            x = self.norm(x)
            x_tar = self.norm(x_tar)
            x_fusion = self.norm(x_fusion)
            x_fusion_src = self.norm(x_fusion_src)
            out_x = self.nn1(self.to_cls_token(x[:, 0]))
            out_x_tar = self.nn1(self.to_cls_token(x_tar[:, 0]))
            out_x_fusion = self.nn1(self.to_cls_token(x_fusion[:, 0]))
            out_fusion_src = self.nn1(self.to_cls_token(x_fusion_src[:, 0]))
            return out_x, out_x_tar, out_x_fusion, out_fusion_src

def AgentBiDA(dataset, opts):
    model = None
    if 'MJG' in dataset.split('_'):
        model = AgentBiDAnet(n_bands=64, num_classes=5,
                             num_tokens=opts.num_tokens, dim=opts.dim, depth=opts.depth, heads=opts.num_heads, num_agents=opts.num_agents)
    elif dataset == 'Houston18' or dataset == 'Houston13':
        model = AgentBiDAnet(n_bands=48, num_classes=7,
                             num_tokens=opts.num_tokens, dim=opts.dim, depth=opts.depth, heads=opts.num_heads, num_agents=opts.num_agents)
    elif dataset == 'Dioni' or dataset == 'Loukia':
        model = AgentBiDAnet(n_bands=176, num_classes=12,
                             num_tokens=opts.num_tokens, dim=opts.dim, depth=opts.depth, heads=opts.num_heads, num_agents=opts.num_agents)
    return model

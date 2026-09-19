"""Factory for implementations actually available in this repository."""
from .BiDA import BiDA
from .GAHT import GAHT


def get_model(model_name, dataset_name, patch_size, opts=None, ema=False):
    if model_name == 'GAHT':
        model = GAHT(dataset_name, patch_size)
    elif model_name == 'BiDA':
        model = BiDA(dataset_name, opts)
    elif model_name == 'AgentBiDA':
        from .agent_bida import AgentBiDA
        model = AgentBiDA(dataset_name, opts)
    elif model_name in ('SelfAttentionAgentBiDA', 'SelfAttnAgentBiDA'):
        from .self_attention_agent_bida import AgentBiDA
        model = AgentBiDA(dataset_name, opts)
    else:
        raise ValueError(f'{model_name} has no available implementation. Use GAHT, BiDA, '
                         'AgentBiDA, or SelfAttentionAgentBiDA.')
    if model is None:
        raise ValueError(f'Unsupported dataset {dataset_name}')
    if ema:
        model.requires_grad_(False)
    return model

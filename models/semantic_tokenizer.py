"""Standalone semantic tokenizer for hyperspectral image (HSI) cubes."""

import torch
from torch import nn


class HSISemanticTokenizer(nn.Module):
    """Convert an HSI cube into a fixed number of semantic tokens.

    The input can be either ``[B, 1, bands, height, width]`` or
    ``[B, bands, height, width]``. The output is ``[B, num_tokens, token_dim]``.

    Each token learns a spatial attention distribution and is the weighted
    average of the feature vectors at all spatial locations. Spectral
    aggregation is adaptive, so the module does not require a fixed number
    of input bands.
    """

    def __init__(
        self,
        num_tokens=4,
        token_dim=64,
        feature_channels=8,
        input_channels=1,
    ):
        super().__init__()
        if num_tokens < 1:
            raise ValueError("num_tokens must be positive")
        if token_dim < 1 or feature_channels < 1:
            raise ValueError("token_dim and feature_channels must be positive")

        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.input_channels = input_channels

        self.spectral_spatial_features = nn.Sequential(
            nn.Conv3d(
                input_channels,
                feature_channels,
                kernel_size=(3, 3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU(inplace=True),
        )

        self.spatial_features = nn.Sequential(
            nn.Conv2d(feature_channels, token_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(token_dim),
            nn.ReLU(inplace=True),
        )

        # Produces one attention logit map for each semantic token.
        self.attention = nn.Conv2d(token_dim, num_tokens, kernel_size=1, bias=False)

    def _as_5d(self, x):
        if x.ndim == 4:
            # [B, bands, H, W] -> [B, 1, bands, H, W]
            x = x.unsqueeze(1)
        elif x.ndim != 5:
            raise ValueError(
                "Expected HSI input with shape [B, bands, H, W] or "
                "[B, channels, bands, H, W]"
            )

        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channel(s), got {x.shape[1]}"
            )
        return x

    def forward(self, x, return_attention=False):
        """Return semantic tokens, optionally together with attention maps."""
        x = self._as_5d(x)
        batch_size, _, _, height, width = x.shape

        x = self.spectral_spatial_features(x)
        # Collapse the spectral axis without assuming a particular band count.
        x = x.mean(dim=2)
        x = self.spatial_features(x)

        # A separate probability distribution over spatial positions per token.
        attention = self.attention(x).flatten(2).softmax(dim=-1)
        values = x.flatten(2)
        tokens = torch.einsum("bln,bcn->blc", attention, values)

        if return_attention:
            attention = attention.view(batch_size, self.num_tokens, height, width)
            return tokens, attention
        return tokens


if __name__ == "__main__":
    tokenizer = HSISemanticTokenizer(num_tokens=4, token_dim=64)
    cubes = torch.randn(2, 48, 13, 13)
    tokens, attention = tokenizer(cubes, return_attention=True)
    print("tokens:", tokens.shape)
    print("attention:", attention.shape)
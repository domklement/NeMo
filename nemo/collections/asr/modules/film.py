import torch
import torch.nn as nn
from typing import Tuple

__all__ = ['FiLM']


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for sequences (B x T x D).

    Args:
        d_model: Feature dimensionality D.
        generator_hidden: Optional hidden size for a tiny MLP FiLM generator.
                          If None, uses a single linear layer (per time step).
        dropout: Dropout applied on (γ, β) predictions.
        zero_init_delta_gamma: If True, initialize Δγ head to zero so γ = 1 at start.
        clamp_gamma: Optional (min, max) to clamp γ for stability. Set to None to disable.
    """
    def __init__(
        self,
        d_model: int,
        generator_hidden: int = None,
        dropout: float = 0.0,
        zero_init_delta_gamma: bool = True,
        clamp_gamma: Tuple[float, float] | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.clamp_gamma = clamp_gamma

        if generator_hidden is None:
            self.gamma_head = nn.Linear(d_model, d_model, bias=True)  # predicts Δγ
            self.beta_head  = nn.Linear(d_model, d_model, bias=True)
        else:
            self.gamma_head = nn.Sequential(
                nn.Linear(d_model, generator_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(generator_hidden, d_model),
            )
            self.beta_head = nn.Sequential(
                nn.Linear(d_model, generator_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(generator_hidden, d_model),
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if zero_init_delta_gamma:
            # Make FiLM an identity at init: γ≈1, β≈0
            def zero_out(m):
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            if isinstance(self.gamma_head, nn.Sequential):
                self.gamma_head.apply(zero_out)
            else:
                zero_out(self.gamma_head)

            if isinstance(self.beta_head, nn.Sequential):
                self.beta_head.apply(zero_out)
            else:
                zero_out(self.beta_head)

    def film_parameters(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute γ and β from conditioning sequence c.

        c: (B, T, D) or (B, D). If (B, D), it's broadcast across time.
        Returns:
            γ, β each shaped (B, T, D)
        """
        if c.dim() == 2:
            c = c.unsqueeze(1)  # (B, 1, D); will broadcast along T when we add/mul
        elif c.dim() != 3:
            raise ValueError("Conditioning input must be (B, T, D) or (B, D)")

        # Apply heads time-step wise (nn.Linear operates on last dim)
        delta_gamma = self.gamma_head(c)  # (B, T, D)
        beta        = self.beta_head(c)   # (B, T, D)

        gamma = 1.0 + delta_gamma
        if self.clamp_gamma is not None:
            gamma = torch.clamp(gamma, self.clamp_gamma[0], self.clamp_gamma[1])

        gamma = self.dropout(gamma)
        beta  = self.dropout(beta)
        return gamma, beta

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: target sequence (B, T, D)
        c: conditioning sequence (B, T, D) or a global vector (B, D)
        """
        if x.dim() != 3:
            raise ValueError("x must be (B, T, D)")
        if c.dim() == 2:
            # Broadcast γ, β across time steps
            gamma, beta = self.film_parameters(c)         # (B, 1, D)
            if gamma.size(1) == 1 and x.size(1) > 1:
                gamma = gamma.expand(-1, x.size(1), -1)
                beta  = beta.expand(-1,  x.size(1), -1)
        else:
            gamma, beta = self.film_parameters(c)         # (B, T, D)
            if gamma.size(1) != x.size(1):
                raise ValueError("Time dimension T must match between x and c when c is (B, T, D)")

        return gamma * x + beta
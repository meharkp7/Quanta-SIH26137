"""
Step 16.13 — PPO Actor-Critic network.

The network consumes the model-independent representation defined in
src.learning.representation.

It does NOT depend on:
    - GNN internals
    - Transformer internals
    - SUMO
    - QPSO
    - route planning

The actor predicts one logit for each ScopeAction.
The critic predicts the scalar state value V(s).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from src.learning.representation import (
    RepresentationOutput,
    RepresentationSpec,
)


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class PPOActorCriticConfig:
    """Configuration for the PPO actor-critic."""

    hidden_dim: int = 256
    hidden_dim_2: int = 128

    action_dim: int = 5

    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        if self.hidden_dim_2 <= 0:
            raise ValueError("hidden_dim_2 must be positive")

        if self.action_dim != 5:
            raise ValueError(
                "action_dim must be exactly 5 for ScopeAction"
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1"
            )


# ============================================================================
# Network
# ============================================================================


class PPOActorCritic(nn.Module):
    """
    Shared-trunk PPO actor-critic.

    Input:
        fixed-size RepresentationOutput or a tensor of shape [B, D]

    Output:
        actor logits: [B, 5]
        critic value: [B, 1]

    The network intentionally treats the representation as an opaque vector.
    """

    def __init__(
        self,
        representation_spec: RepresentationSpec,
        config: PPOActorCriticConfig | None = None,
    ) -> None:
        super().__init__()

        self.representation_spec = representation_spec
        self.config = config or PPOActorCriticConfig()

        input_dim = representation_spec.total_dim

        self.shared = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.Tanh(),
            nn.Linear(
                self.config.hidden_dim,
                self.config.hidden_dim_2,
            ),
            nn.LayerNorm(self.config.hidden_dim_2),
            nn.Tanh(),
            nn.Dropout(self.config.dropout),
        )

        self.actor = nn.Linear(
            self.config.hidden_dim_2,
            self.config.action_dim,
        )

        self.critic = nn.Linear(
            self.config.hidden_dim_2,
            1,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        PPO-friendly initialization.

        Small actor output weights prevent extremely confident initial
        policies. The critic starts near zero.
        """

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _representation_to_tensor(
        self,
        representation: RepresentationOutput | Tensor | np.ndarray,
    ) -> Tensor:
        if isinstance(representation, RepresentationOutput):
            if representation.spec != self.representation_spec:
                raise ValueError(
                    "RepresentationSpec does not match the "
                    "actor-critic specification"
                )

            tensor = torch.as_tensor(
                representation.flat.copy(),
                dtype=torch.float32,
            )

        elif isinstance(representation, np.ndarray):
            tensor = torch.as_tensor(
                representation,
                dtype=torch.float32,
            )

        elif isinstance(representation, Tensor):
            tensor = representation

        else:
            raise TypeError(
                "representation must be RepresentationOutput, "
                "numpy.ndarray, or torch.Tensor"
            )

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim != 2:
            raise ValueError(
                f"representation must have shape [B, D], "
                f"got {tuple(tensor.shape)}"
            )

        if tensor.shape[-1] != self.representation_spec.total_dim:
            raise ValueError(
                "representation dimension mismatch: "
                f"expected {self.representation_spec.total_dim}, "
                f"got {tensor.shape[-1]}"
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                "representation contains NaN or infinite values"
            )

        # Representations are produced by the environment/state pipeline on
        # CPU (NumPy). The actor-critic owns the device boundary, so every
        # representation entering the network must be moved to the same
        # device as the model parameters. This is required for MPS/CUDA as
        # well as CPU and keeps callers device-agnostic.
        model_device = next(self.parameters()).device
        return tensor.to(device=model_device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        representation: RepresentationOutput | Tensor | np.ndarray,
        action_mask: Tensor | np.ndarray | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Return actor logits and critic values.

        Parameters
        ----------
        representation:
            Fixed-size PPO representation.

        action_mask:
            Optional boolean/binary mask of shape [5] or [B, 5].

            True / 1  -> action is allowed
            False / 0 -> action is unavailable

        Returns
        -------
        logits:
            Tensor [B, 5]

        values:
            Tensor [B, 1]
        """

        x = self._representation_to_tensor(representation)

        shared = self.shared(x)

        logits = self.actor(shared)
        values = self.critic(shared)

        if action_mask is not None:
            logits = self._apply_action_mask(
                logits,
                action_mask,
            )

        return logits, values

    # ------------------------------------------------------------------
    # Action masking
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_action_mask(
        logits: Tensor,
        action_mask: Tensor | np.ndarray,
    ) -> Tensor:
        """
        Apply a feasibility mask to actor logits.

        Invalid actions receive a very large negative logit so that their
        probability becomes effectively zero.

        At least one action must remain feasible for every batch element.
        """

        if isinstance(action_mask, np.ndarray):
            mask = torch.as_tensor(
                action_mask,
                dtype=torch.bool,
                device=logits.device,
            )
        else:
            mask = action_mask.to(
                device=logits.device,
                dtype=torch.bool,
            )

        if mask.ndim == 1:
            mask = mask.unsqueeze(0)

        if mask.shape != logits.shape:
            raise ValueError(
                "action_mask must have shape "
                f"{tuple(logits.shape)}, got {tuple(mask.shape)}"
            )

        if not mask.any(dim=-1).all():
            raise ValueError(
                "Every batch element must have at least one "
                "feasible action"
            )

        return logits.masked_fill(
            ~mask,
            torch.finfo(logits.dtype).min,
        )

    # ------------------------------------------------------------------
    # Distribution / action selection
    # ------------------------------------------------------------------

    def distribution(
        self,
        representation: RepresentationOutput | Tensor | np.ndarray,
        action_mask: Tensor | np.ndarray | None = None,
    ) -> Categorical:
        """Return the categorical policy distribution."""

        logits, _ = self.forward(
            representation,
            action_mask=action_mask,
        )

        return Categorical(logits=logits)

    def sample_action(
        self,
        representation: RepresentationOutput | Tensor | np.ndarray,
        action_mask: Tensor | np.ndarray | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Sample an action from the policy.

        Returns:
            action: [B]
            log_probability: [B]
            value: [B]
        """

        logits, values = self.forward(
            representation,
            action_mask=action_mask,
        )

        distribution = Categorical(logits=logits)

        action = distribution.sample()
        log_probability = distribution.log_prob(action)

        return (
            action,
            log_probability,
            values.squeeze(-1),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        representation: RepresentationOutput | Tensor | np.ndarray,
        actions: Tensor,
        action_mask: Tensor | np.ndarray | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate actions during PPO optimization.

        Returns:
            log_probability: [B]
            entropy: [B]
            values: [B]
        """

        distribution = self.distribution(
            representation,
            action_mask=action_mask,
        )

        if actions.ndim != 1:
            actions = actions.reshape(-1)

        actions = actions.to(
            device=distribution.logits.device,
            dtype=torch.long,
        )

        log_probability = distribution.log_prob(actions)
        entropy = distribution.entropy()

        _, values = self.forward(
            representation,
            action_mask=action_mask,
        )

        return (
            log_probability,
            entropy,
            values.squeeze(-1),
        )
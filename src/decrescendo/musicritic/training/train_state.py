"""Training state management using Flax."""

from typing import Any, Callable

import jax
import optax
from flax.training import train_state


class TrainState(train_state.TrainState):
    """Extended TrainState with dropout RNG key.

    Attributes:
        dropout_rng: JAX random key for dropout during training
    """

    dropout_rng: jax.Array | None = None

    @classmethod
    def create(
        cls,
        *,
        apply_fn: Callable[..., Any],
        params: dict[str, Any],
        tx: optax.GradientTransformation,
        dropout_rng: jax.Array | None = None,
        **kwargs: Any,
    ) -> "TrainState":
        """Create a new TrainState.

        Args:
            apply_fn: Model apply function
            params: Model parameters
            tx: Optax optimizer/gradient transformation
            dropout_rng: Optional random key for dropout
            **kwargs: Additional arguments passed to parent

        Returns:
            Initialized TrainState
        """
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            dropout_rng=dropout_rng,
            **kwargs,
        )

    def next_dropout_rng(self) -> tuple["TrainState", jax.Array]:
        """Split the dropout RNG and return new state with updated RNG.

        Returns:
            Tuple of (new_state, rng_for_this_step)

        Raises:
            ValueError: If dropout_rng was not set during creation
        """
        if self.dropout_rng is None:
            raise ValueError("dropout_rng not set in TrainState")

        current_rng, new_rng = jax.random.split(self.dropout_rng)
        new_state = self.replace(dropout_rng=new_rng)
        return new_state, current_rng

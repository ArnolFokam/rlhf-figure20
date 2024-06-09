from typing import NamedTuple

import flax
import jax.numpy as jnp
import flax.linen as nn

# NamedTuple for preference model parameters
class LMBackboneWithScalarHeadParams(NamedTuple):
    backbone_params: flax.core.FrozenDict
    head_params: flax.core.FrozenDict

# Regression head class for preference model
class RegressionHead(nn.Module):
    head_input_size: int
    param_dtype: jnp.dtype

    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, dtype=self.param_dtype)(x)

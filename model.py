import flax.linen as nn

# Regression head class for preference model
class RegressionHead(nn.Module):
    head_input_size: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(1)(x)

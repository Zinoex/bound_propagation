# Bound propagation

Linear and interval bound propagation for PyTorch, with an ergonomic API and GPU support.

To install:
```
pip install bound-propagation
```

## Constructing a `BoundModel` and computing bounds

The user-facing API is `BoundModel`. It traces the model once at construction time, runs a metadata pass against `dummy_inputs`, and builds a propagator for the chosen `method`. Subsequent `propagate` calls reuse the traced graph.

```python
import torch
from torch import nn
from bound_propagation import BoundModel, HyperRectangle

class Network(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(30, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 10),
        )

net = Network()
dummy = torch.zeros(30)                  # one tensor per placeholder, feature-shaped (no batch dimension!)

bm = BoundModel(net, dummy_inputs=(dummy,), method="backward_lbp")

x = torch.rand(100, 30)
region = HyperRectangle.from_eps(x, 0.1) # leading dim `100` is batch
bounds = bm.propagate(region)            # -> LinearBounds
lower, upper = bounds.concretize()       # -> IntervalBounds, then unpack
```

Batch dimensions are inferred per call: any leading dims on a region beyond the dummy's feature shape are treated as batch. All input regions must agree on the batch shape.

`BoundModel` takes one of five `method` values, and the returned bound type follows:

| `method`                | Returns                        | Notes                                             |
|-------------------------|--------------------------------|---------------------------------------------------|
| `"ibp"`                 | `IntervalBounds`               | Interval arithmetic; fast, least precise.         |
| `"forward_lbp"`         | `LinearBounds`                 | Affine bounds propagated forward.                 |
| `"backward_lbp"`        | `LinearBounds`                 | CROWN: recursive backward concretization.         |
| `"forward_backward_lbp"`| `LinearBounds`                 | Forward LBP for intermediates, backward for output. |
| `"crown_ibp"`           | `LinearBounds`                 | IBP for intermediates, backward for output.       |

Call `.concretize()` on either bound type to obtain an `IntervalBounds` with `lower` / `upper` tensors (it's a no-op for IBP).

### α-CROWN

Each non-linear activation's relaxation has a free slope parameter (e.g. ReLU's lower-bound slope α ∈ [0, 1]) that's sound for any value but optimal at different values depending on how downstream layers reweight the unit. α-CROWN ([Xu et al., 2021](https://arxiv.org/abs/2011.13824)) tunes those parameters via projected gradient descent on the output bounds, usually giving tighter bounds than plain CROWN at the cost of extra passes.

```python
from bound_propagation import AlphaOptimizationConfig

cfg = AlphaOptimizationConfig(enabled=True, num_steps=20, lr=0.1)
bm = BoundModel(net, dummy_inputs=(dummy,), method="backward_lbp", alpha=cfg)
```

`optimize_intermediate=True` also couples intermediate bounds' α parameters (tighter, more expensive) and is only valid for `"backward_lbp"`.

### Registering a custom operation

`RegistryExtension` bundles strategies for one or more `fx` targets (`nn.Module` classes or free Python functions). Supply strategies for whichever registry key(s) your method needs (see `BoundModel.required_registry_keys`).

```python
from bound_propagation import BoundModel, RegistryExtension

ext = RegistryExtension(
    targets=[my_fn, MyModule],            # free functions and/or nn.Module classes
    ibp=MyIBPStrategy(),
    forward_lbp=MyForwardLBPStrategy(),
    backward_lbp=MyBackwardLBPStrategy(),
)
bm = BoundModel(net, dummy_inputs=(dummy,), method="crown_ibp", extensions=[ext])
```

### Graph simplification

Passing `simplify=True` (default) runs `SimplificationPass` on the traced graph before propagation, folding algebraic identities (e.g. `x * x → pow(x, 2)`) and dropping structural no-ops. These simplifications often result in tighter  Note that some rewrites can introduce targets the chosen registry must support.

## Interpreting `LinearBounds`

`LinearBounds` represents affine relaxations of the output w.r.t. the model's inputs:

```
bias_lower + Σ_i W_lower_i · x_i   ≤   y   ≤   bias_upper + Σ_i W_upper_i · x_i
```

where each `x_i` ranges over the `i`-th input region and `W_{lower,upper}_i` is an element of `LinearOperator` — an abstract linear map from input axes to output axes. The fields on `LinearBounds` are:

- `bias_lower`, `bias_upper` — dense tensors of shape `(*batch_dims, *output_shape)`.
- `regions` — list of `SimpleRegion`, one per input contributing to the affine form.
- `input_ids` — list of input node IDs; stable identifiers for the input a given term is affine in (relevant with multiple inputs).
- `linear_lowers_op`, `linear_uppers_op` — lists of `LinearOperator` instances, parallel to `regions` / `input_ids`.
- `linear_lowers`, `linear_uppers` — convenience accessors that materialize each operator to a dense coefficient tensor of shape `(*batch_dims, *output_shape, *input_shape)` via `LinearOperator.to_dense()`.

When the bounds are with respect to a single input, the following singular accessors are available:
- `linear_lower`
- `linear_upper`
- `linear_lower_op`
- `linear_upper_op`
- `region`
- `input_id`

### `LinearOperator` types

Each entry of `linear_lowers_op` / `linear_uppers_op` is a `LinearOperator`. The shape convention matches `LinearBounds`:

```
output_shape = (*batch_dims, *output_dims)   # matches bias tensor shape
input_shape  = (*input_dims,)                # trailing axes describing x
```

Operators expose `apply(x)` / `apply_transpose(y)` for linear maps, and `concretize_min(region)` / `concretize_max(region)` for evaluating the min/max contribution over a region — this is what `LinearBounds.concretize()` uses.

Currently shipped:

- **`DenseOperator`** — wraps a tensor of shape `(*output_shape, *input_shape)`. This is the universal fallback and is what you'll see from every built-in strategy today. `linear_lowers` / `linear_uppers` return the underlying dense tensors directly.

Structured operators (e.g. convolution / pooling that carry their algebraic structure through the pipeline without materializing) are planned; the `LinearOperator` interface is designed so those can be added without changing the `LinearBounds` API or user code. Until then, convolution and pooling strategies produce `DenseOperator` coefficients.

### Example: reading bounds out

```python
bm = BoundModel(net, dummy_inputs=(dummy,), method="backward_lbp")
bounds = bm.propagate(region)

# Concrete interval — easiest form to consume.
lower, upper = bounds.concretize()

# Affine form, single-input case.
W_lower = bounds.linear_lower          # dense tensor, shape (*batch, *out, *in)
b_lower = bounds.bias_lower            # dense tensor, shape (*batch, *out)
W_upper = bounds.linear_upper
b_upper = bounds.bias_upper

# Check bounds.has_linear_terms() first if the model might produce purely
# constant bounds (no dependence on inputs).
```

## Design philosophy

`bound_propagation` traces a PyTorch model with [`torch.fx`](https://pytorch.org/docs/stable/fx.html) into a computation graph of primitive operations, then walks that graph dispatching each node to a per-operation bounding *strategy* selected by a `TargetRegistry` (Factory pattern). Each strategy lives in its own file and implements the math for one op under one method (IBP, forward LBP, backward LBP) — see `src/bound_propagation/propagation/{ibp,forward_lbp,backward_lbp}/`.

This makes the library relative to [auto_LiRPA](https://github.com/KaidiXu/auto_LiRPA) a different set of tradeoffs rather than a strict replacement:

- **Graph source.** We trace `nn.Module` / plain callables via `torch.fx` on the Python side; auto_LiRPA parses ONNX. Tracing `torch.fx` means we stay closer to user code and error messages point back to Python operations.
- **Extensibility.** Adding a new op is "write one strategy class per method you care about, then register it on a `RegistryExtension`." There is no central dispatch tree to edit, and users can register strategies for their own custom `nn.Module`s or functions without forking the library.
- **Supported operations.** Out of the box: arithmetic (`+`, `-`, `*`, `/`, `@`, `neg`, `pow`), element-wise activations (`ReLU`, `Sigmoid`, `Tanh`, `Exp`, `Log`, `Sqrt`, `Reciprocal`, `Abs`, `Clamp`, `Sin`, `Cos`, `Tan`), `Linear`, reductions (`sum`, `mean`, `amax`, `amin`), shape ops (`reshape`, `view`, `flatten`, `cat`, `stack`, `getitem`, `select`, `squeeze`, `unsqueeze`, `transpose`, `permute`), and `torch.maximum`/`torch.minimum`. The exact targets and their strategies are listed in `propagation/{ibp,forward_lbp,backward_lbp}/__init__.py`.
- **Propagation methods.** IBP, forward LBP, backward LBP (CROWN), forward-backward LBP, and CROWN-IBP, each with optional α-CROWN parameter optimization. Methods are orthogonal to operations — a user-provided strategy slots into whichever method(s) it implements.

Supported propagation methods:
- Interval Bound Propagation (IBP)
- Forward Linear Bound Propagation (Forward LBP)
- Backward Linear Bound Propagation / [CROWN](https://arxiv.org/abs/1811.00866)
- Forward–Backward Linear Bound Propagation (forward LBP with a final backward pass)
- [CROWN-IBP](https://arxiv.org/abs/1906.06316)

All LBP-based methods accept an `AlphaOptimizationConfig` to enable α-CROWN (optimizing per-relaxation slope parameters via projected gradient descent).

## Authors
- [Frederik Baymler Mathiesen](https://www.baymler.com) — PhD student @ TU Delft

## Citing
```
@misc{Mathiesen2022,
  author = {Frederik Baymler Mathiesen},
  title = {Bound Propagation},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Zinoex/bound_propagation}}
}
```

## Funding and support
- TU Delft

## Copyright notice:
Technische Universiteit Delft hereby disclaims all copyright interest in the program “bound_propagation” (bound propagation methods for Pytorch) written by the Frederik Baymler Mathiesen. Theun Baller, Dean of Mechanical, Maritime and Materials Engineering

© 2026, Frederik Baymler Mathiesen, HERALD Lab, TU Delft

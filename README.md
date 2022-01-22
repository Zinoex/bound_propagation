# Bound propagation
Linear and interval bound propagation in Pytorch with easy-to-use API, GPU support, and heavy parallization.
Initially made as an alternative to [the original CROWN implementation](https://github.com/IBM/CROWN-Robustness-Certification) which featured only Numpy, lots of for-loops, and a cumbersome API.

To install:
```
pip install bound-propagation
```

Supported bound propagation methods:
- Interval Bound Propagation (IBP)
- [CROWN](https://arxiv.org/abs/1811.00866)
- [CROWN-IBP](https://arxiv.org/abs/1906.06316)

For the examples below assume the following network definition:
```python
from torch import nn
from bound_propagation import crown, crown_ibp, ibp

# The decorators _must_ be on a subclass of nn.Sequential

@crown
@crown_ibp
@ibp
class Network(nn.Sequential):
    def __init__(self, *args):
        if args:
            # To support __get_index__ of nn.Sequential when slice indexing
            # CROWN (and implicitly CROWN-IBP) is doing this underlying
            super().__init__(*args)
        else:
            in_size = 30
            classes = 10

            super().__init__(
                nn.Linear(in_size, 16),
                nn.Tanh(),
                nn.Linear(16, 16),
                nn.Tanh(),
                nn.Linear(16, classes)
            )

net = Network()
```

Alternatively, you can add the functions to your network by calling the functions with an instance of your network:
```python
from torch import nn
from bound_propagation import crown, crown_ibp, ibp

class Network(nn.Sequential):
    def __init__(self, *args):
        if args:
            # To support __get_index__ of nn.Sequential when slice indexing
            # CROWN (and implicitly CROWN-IBP) is doing this underlying
            super().__init__(*args)
        else:
            in_size = 30
            classes = 10

            super().__init__(
                nn.Linear(in_size, 16),
                nn.Tanh(),
                nn.Linear(16, 16),
                nn.Tanh(),
                nn.Linear(16, classes)
            )

# The instance _must_ be an nn.Sequential or a subclass thereof
net = crown(crown_ibp(ibp(Network())))
```

The method also works with ```nn.Sigmoid``` and ```nn.ReLU```.

## Interval bounds
To get interval bounds for either IBP, CROWN, or CROWN-IBP:

```python
x = torch.rand(100, 30)
epsilon = 0.1
lower, upper = x - epsilon, x + epsilon

ibp_bounds = net.ibp(lower, upper)
crown_bounds = net.crown_interval(lower, upper)
crown_ibp_bounds = net.crown_ibp_interval(lower, upper)
```

## Linear bounds
To get linear bounds for either CROWN or CROWN-IBP:

```python
x = torch.rand(100, 30)
epsilon = 0.1
lower, upper = x - epsilon, x + epsilon

crown_bounds = net.crown_linear(lower, upper)
crown_ibp_bounds = net.crown_ibp_linear(lower, upper)
```

## Authors
- [Frederik Baymler Mathiesen](https://www.baymler.com) - PhD student @ TU Delft
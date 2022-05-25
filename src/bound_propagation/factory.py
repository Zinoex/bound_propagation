from torch import nn

from .activation import BoundReLU, BoundTanh, BoundSigmoid, BoundIdentity
from .bivariate import BoundAdd, Add, BoundSub, Sub, VectorAdd, BoundVectorAdd, VectorSub, BoundVectorSub
from .cat import BoundCat, Cat
from .linear import BoundLinear
from .parallel import BoundParallel, Parallel
from .residual import BoundResidual, Residual
from .saturation import BoundClamp, Clamp
from .sequential import BoundSequential


class BoundModelFactory:
    """
    This class is a combination of the factory and visitor pattern to construct the bound module graph recursively while
    being flexible to add new types.
    As an example, say you design a new nn.Module subtype and let that be a layer in a nn.Sequential. With the factory,
    you can add the new subtype to the factory, and when constructing BoundSequential, this will be recognized.
    """
    def __init__(self, **kwargs):
        self.class_mapping = [
            (nn.Sequential, BoundSequential),
            (nn.Linear, BoundLinear),
            (nn.ReLU, BoundReLU),
            (nn.Tanh, BoundTanh),
            (nn.Sigmoid, BoundSigmoid),
            (nn.Identity, BoundIdentity),
            (Add, BoundAdd),
            (VectorAdd, BoundVectorAdd),
            (Sub, BoundSub),
            (VectorSub, BoundVectorSub),
            (Residual, BoundResidual),
            (Cat, BoundCat),
            (Parallel, BoundParallel),
            (Clamp, BoundClamp)
        ]

        self.kwargs = kwargs

    def register(self, original_class, bound_class):
        self.class_mapping.insert(0, (original_class, bound_class))

    def build(self, module):
        for orig_class, bound_class in self.class_mapping:
            if isinstance(module, orig_class):
                return bound_class(module, self, **self.kwargs)

        raise NotImplementedError('Module type not supported - add BoundModule for layer to factory')

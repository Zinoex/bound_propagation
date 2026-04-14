import torch

from bound_propagation.bounds import LinearBounds


def forward_compose(self, other: LinearBounds) -> LinearBounds:
    """
    Forward compose these bounds with another set of linear bounds.

    This is used to propagate bounds through relaxed non-linear operations:
    if we have linear bounds `other` on an operation (input/output relation),
    we can compose them with linear bounds on the input to get linear bounds
    of the composition of `self` and this non-linear operation wrt. the input.

    Mathematically, if we call `self` $f$ and `other` $g$, then this represents $g \\circ f$.
    The input region of `self` correspond to the global input region while `other`'s input region
    is local to its node; we should check whether the bounds are compatible before composing.
    """
    # Substitute self's bounds into other's bounds
    # self: y = W_l^f @ x + b_l^f (lower), y = W_u^f @ x + b_u^f (upper)
    # other: z = W_l^g @ y + b_l^g (lower), z = W_u^g @ y + b_u^g (upper)
    # Result: z in terms of x

    # For lower bound: W_l^g @ y + b_l^g
    # Use positive weights with lower bound, negative weights with upper bound
    if other.linear_lower is not None:
        weights_lower_pos = torch.clamp(other.linear_lower, min=0)
        weights_lower_neg = torch.clamp(other.linear_lower, max=0)

        # Linear coefficients
        if self.linear_lower is not None and self.linear_upper is not None:
            linear_lower = weights_lower_pos @ self.linear_lower + weights_lower_neg @ self.linear_upper
        else:
            linear_lower = None

        # Bias
        bias_lower = weights_lower_pos @ self.bias_lower + weights_lower_neg @ self.bias_upper + other.bias_lower
    else:
        linear_lower = None
        bias_lower = other.bias_lower

    # For upper bound: W_u^g @ y + b_u^g
    # Use positive weights with upper bound, negative weights with lower bound
    if other.linear_upper is not None:
        weights_upper_pos = torch.clamp(other.linear_upper, min=0)
        weights_upper_neg = torch.clamp(other.linear_upper, max=0)

        # Linear coefficients
        if self.linear_lower is not None and self.linear_upper is not None:
            linear_upper = weights_upper_pos @ self.linear_upper + weights_upper_neg @ self.linear_lower
        else:
            linear_upper = None

        # Bias
        bias_upper = weights_upper_pos @ self.bias_upper + weights_upper_neg @ self.bias_lower + other.bias_upper
    else:
        linear_upper = None
        bias_upper = other.bias_upper

    return LinearBounds(
        regions=self.regions,
        linear_lower=linear_lower,
        bias_lower=bias_lower,
        linear_upper=linear_upper,
        bias_upper=bias_upper,
    )


def backward_compose(self, other: LinearBounds) -> LinearBounds:
    """
    Backward compose these bounds with another set of linear bounds.

    This is used to propagate bounds backwards through relaxed non-linear operations:
    if we have linear bounds `other` on an operation (input/output relation),
    we can compose them with linear bounds on the output to get linear bounds
    of the composition of this non-linear operation and `self` wrt. the output.

    Mathematically, if we call `self` $f$ and `other` $g$, then this represents $f \\circ g$.
    The input region of `self` correspond to input region over which the relaxation of $f$ is valid,
    while `other`'s input region is local to its node; we should check whether
    the bounds are compatible before composing.
    """
    # Substitute other's bounds into self's bounds
    # other: y = W_l^g @ x + b_l^g (lower), y = W_u^g @ x + b_u^g (upper)
    # self: z = W_l^f @ y + b_l^f (lower), z = W_u^f @ y + b_u^f (upper)
    # Result: z in terms of x

    # For lower bound: W_l^f @ y + b_l^f
    # Use positive weights with lower bound, negative weights with upper bound
    if self.linear_lower is not None:
        weights_lower_pos = torch.clamp(self.linear_lower, min=0)
        weights_lower_neg = torch.clamp(self.linear_lower, max=0)

        # Linear coefficients
        if other.linear_lower is not None and other.linear_upper is not None:
            linear_lower = weights_lower_pos @ other.linear_lower + weights_lower_neg @ other.linear_upper
        else:
            linear_lower = None

        # Bias
        bias_lower = weights_lower_pos @ other.bias_lower + weights_lower_neg @ other.bias_upper + self.bias_lower
    else:
        linear_lower = None
        bias_lower = self.bias_lower

    # For upper bound: W_u^f @ y + b_u^f
    # Use positive weights with upper bound, negative weights with lower bound
    if self.linear_upper is not None:
        weights_upper_pos = torch.clamp(self.linear_upper, min=0)
        weights_upper_neg = torch.clamp(self.linear_upper, max=0)

        # Linear coefficients
        if other.linear_lower is not None and other.linear_upper is not None:
            linear_upper = weights_upper_pos @ other.linear_upper + weights_upper_neg @ other.linear_lower
        else:
            linear_upper = None

        # Bias
        bias_upper = weights_upper_pos @ other.bias_upper + weights_upper_neg @ other.bias_lower + self.bias_upper
    else:
        linear_upper = None
        bias_upper = self.bias_upper

    return LinearBounds(
        regions=other.regions,
        linear_lower=linear_lower,
        bias_lower=bias_lower,
        linear_upper=linear_upper,
        bias_upper=bias_upper,
    )

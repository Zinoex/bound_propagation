from torch import nn


class UnivariateMonomial(nn.Module):
    def __init__(self, powers):
        super().__init__()

        # Assume powers are of the structure [(index, power)] and this will be the output order too
        self.powers = powers

    def forward(self, x):
        x = [x[..., index] ** power for index, power in self.powers]
        return x


class MultivariateMonomial(nn.Module):
    pass

    # Combine Mul and UnivariateMonomial to support multivariate monomials


class Polynomials(nn.Module):
    pass

    # Combine Linear and MultivariateMonomial to form Polynomials

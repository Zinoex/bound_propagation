# TODO: Passes to replace challenging patterns with simpler equivalents.
# Prioritize patterns that are common in practice and that lead to tighter relaxations, e.g.
# if a composition of two non-linearities can be replaced with a single non-linearity + linear operations,
# that will generally yield a tighter relaxation than the composition of two relaxations.

# Polynomial simplications:
# - x * x → x²

# Trigonometric identities:
# - cos(x) * sin(x) → 0.5 * sin(2x)
# - cos²(x) → 0.5 * (1 + cos(2x))
# - sin²(x) → 0.5 * (1 - cos(2x))

# Exponentials and logarithms:
# - exp(x) * exp(y) → exp(x + y)
# - log(exp(x)) → x

# Linear factoring:
# - x * y + x * z → x * (y + z)
# - x * y + z * y → (x + z) * y
# - x * y - x * z → x * (y - z)
# - x * y - z * y → (x - z) * y


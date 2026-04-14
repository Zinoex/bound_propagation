from __future__ import annotations

from typing import Literal

import torch
from plum import dispatch

from ..regions import AbstractRegion, HyperRectangle, SimpleRegion
from .abstract_bounds import AbstractBounds


class LinearBounds(AbstractBounds):
    """
    Linear bounds using affine relaxations.

    Represents bounds as affine functions: lower = W_l @ x + b_l, upper = W_u @ x + b_u
    This allows for tighter bounds through linear relaxations of non-linear operations.

    Used in LBP-style bound propagation methods.

    Attributes:
        regions: Input regions defining the domain of each affine term
        linear_lower: Linear coefficients for lower bound (W_l), one tensor per input region
        bias_lower: Bias term for lower bound (b_l)
        linear_upper: Linear coefficients for upper bound (W_u), one tensor per input region
        bias_upper: Bias term for upper bound (b_u)
        input_ids: Optional list of input node IDs that contribute to these bounds.
                   Used for tracking dependencies in multi-input scenarios.
    """

    def __init__(
        self,
        regions: SimpleRegion | list[SimpleRegion],
        linear_lower: torch.Tensor | list[torch.Tensor] | None = None,
        bias_lower: torch.Tensor | None = None,
        linear_upper: torch.Tensor | list[torch.Tensor] | None = None,
        bias_upper: torch.Tensor | None = None,
        input_ids: list[int] | None = None,
    ) -> None:
        """
        Initialize linear bounds.

        Args:
            regions: Input regions, one for each affine coefficient tensor
            linear_lower: Linear coefficients for lower bound (can be empty for constant bounds)
            bias_lower: Bias for lower bound
            linear_upper: Linear coefficients for upper bound (can be empty for constant bounds)
            bias_upper: Bias for upper bound
        """
        if bias_lower is None or bias_upper is None:
            raise ValueError("LinearBounds requires both bias_lower and bias_upper")

        normalized_regions = self._normalize_regions(regions)
        normalized_linear_lower = self._normalize_linear_terms(linear_lower, "linear_lower")
        normalized_linear_upper = self._normalize_linear_terms(linear_upper, "linear_upper")

        self._check_uniformity(normalized_regions, normalized_linear_lower, normalized_linear_upper)

        if normalized_linear_lower:
            normalized_input_ids = self._normalize_input_ids(input_ids, normalized_regions)
        else:
            normalized_regions = []
            normalized_input_ids = []

        self._check_shapes(normalized_regions, normalized_linear_lower, bias_lower, normalized_linear_upper, bias_upper)
        self._check_gap(normalized_regions, normalized_linear_lower, bias_lower, normalized_linear_upper, bias_upper)

        self._regions = normalized_regions
        self._linear_lower = normalized_linear_lower
        self._bias_lower = bias_lower
        self._linear_upper = normalized_linear_upper
        self._bias_upper = bias_upper
        self._input_ids = normalized_input_ids

    @staticmethod
    def _normalize_regions(regions: SimpleRegion | list[SimpleRegion]) -> list[SimpleRegion]:
        if isinstance(regions, SimpleRegion):
            return [regions]

        normalized_regions = list(regions)
        if any(not isinstance(region, SimpleRegion) for region in normalized_regions):
            raise TypeError("regions must contain only SimpleRegion instances")
        return normalized_regions

    @staticmethod
    def _normalize_linear_terms(
        linear_terms: torch.Tensor | list[torch.Tensor] | None,
        name: str,
    ) -> list[torch.Tensor]:
        if linear_terms is None:
            return []
        if isinstance(linear_terms, torch.Tensor):
            return [linear_terms]

        normalized_terms = list(linear_terms)
        if any(not isinstance(linear, torch.Tensor) for linear in normalized_terms):
            raise TypeError(f"{name} must contain only torch.Tensor entries")
        return normalized_terms

    @staticmethod
    def _normalize_input_ids(input_ids: list[int] | None, regions: list[SimpleRegion]) -> list[int]:
        normalized_input_ids = list(input_ids) if input_ids is not None else [id(region) for region in regions]

        if len(normalized_input_ids) != len(regions):
            raise ValueError(
                f"input_ids must have the same length as regions: {len(normalized_input_ids)} vs {len(regions)}"
            )
        if len(set(normalized_input_ids)) != len(normalized_input_ids):
            raise ValueError(f"input_ids must be unique, but got {normalized_input_ids!r}")

        return normalized_input_ids

    @staticmethod
    def _split_region_shape(
        region_shape: torch.Size,
        output_shape: torch.Size,
        linear_input_axes: torch.Size,
    ) -> tuple[torch.Size, torch.Size]:
        """Infer (*batch_dims, *input_dims) from linear/bias ranks."""
        input_ndim = len(linear_input_axes)
        batch_ndim = len(region_shape) - input_ndim

        if batch_ndim < 0:
            raise ValueError(
                f"Region rank {len(region_shape)} is smaller than inferred input rank {input_ndim} "
                f"from linear input axes {tuple(linear_input_axes)}"
            )

        if batch_ndim > len(output_shape):
            raise ValueError(
                f"Inferred batch rank {batch_ndim} exceeds output rank {len(output_shape)} "
                f"for region shape {tuple(region_shape)} and output shape {tuple(output_shape)}"
            )

        return region_shape[:batch_ndim], region_shape[batch_ndim:]

    def _check_shapes(
        self,
        regions: list[SimpleRegion],
        linear_lower: list[torch.Tensor],
        bias_lower: torch.Tensor,
        linear_upper: list[torch.Tensor],
        bias_upper: torch.Tensor,
    ) -> None:
        if bias_lower.shape != bias_upper.shape:
            raise ValueError(
                f"bias_lower and bias_upper must have the same shape: {bias_lower.shape} vs {bias_upper.shape}"
            )

        for name, linears in (("linear_lower", linear_lower), ("linear_upper", linear_upper)):
            for linear, region in zip(linears, regions, strict=True):
                if len(linear.shape) < len(bias_lower.shape):
                    raise ValueError(
                        f"{name} must include output axes matching bias shape {bias_lower.shape}, got {linear.shape}"
                    )

                output_shape = linear.shape[: len(bias_lower.shape)]
                if output_shape != bias_lower.shape:
                    raise ValueError(f"{name} output shape must match bias shape: {output_shape} vs {bias_lower.shape}")

                input_axes = linear.shape[len(bias_lower.shape) :]
                region_shape = torch.Size(region.shape)
                _, input_shape = self._split_region_shape(region_shape, bias_lower.shape, input_axes)
                if input_axes != input_shape:
                    raise ValueError(
                        f"{name} input axes must match input shape {tuple(input_shape)} "
                        f"(derived from region shape {tuple(region_shape)} and bias shape {tuple(bias_lower.shape)}), "
                        f"got {tuple(input_axes)}"
                    )

    @staticmethod
    def _check_uniformity(
        regions: list[SimpleRegion],
        linear_lower: list[torch.Tensor],
        linear_upper: list[torch.Tensor],
    ) -> None:
        if bool(linear_lower) != bool(linear_upper):
            raise ValueError("linear_lower and linear_upper must either both be provided or both be empty")

        if linear_lower and len(linear_lower) != len(regions):
            raise ValueError(
                f"linear_lower must have the same length as regions: {len(linear_lower)} vs {len(regions)}"
            )

        if linear_upper and len(linear_upper) != len(regions):
            raise ValueError(
                f"linear_upper must have the same length as regions: {len(linear_upper)} vs {len(regions)}"
            )

    def _check_gap(
        self,
        regions: list[SimpleRegion],
        linear_lower: list[torch.Tensor],
        bias_lower: torch.Tensor,
        linear_upper: list[torch.Tensor],
        bias_upper: torch.Tensor,
    ) -> None:
        min_gap = bias_upper - bias_lower

        for region, lower_linear, upper_linear in zip(regions, linear_lower, linear_upper, strict=True):
            min_gap = min_gap + self._minimize_affine_term(region, upper_linear - lower_linear, bias_lower.shape)

        if torch.any(min_gap < -1e-6):
            num_violations = torch.sum(min_gap < -1e-6).item()
            raise ValueError(f"Invalid bounds: upper bound is less than lower bound for {num_violations} outputs")

    @staticmethod
    @dispatch
    def _minimize_affine_term(region: SimpleRegion, linear: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
        raise NotImplementedError(f"Concretization is not implemented for region type {type(region).__name__}")

    @staticmethod
    @dispatch
    def _minimize_affine_term(region: HyperRectangle, linear: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:  # noqa: F811
        input_lower = region.lower
        input_upper = region.upper

        region_shape = torch.Size(region.shape)
        linear_input_axes = torch.Size(linear.shape[len(output_shape) :])
        batch_shape, input_shape = LinearBounds._split_region_shape(region_shape, output_shape, linear_input_axes)
        input_ndim = len(input_shape)
        output_ndim = len(output_shape)

        if linear_input_axes == input_shape:
            expanded_shape = (*batch_shape, *([1] * (output_ndim - len(batch_shape))), *input_shape)
            expanded_lower = input_lower.reshape(expanded_shape)
            expanded_upper = input_upper.reshape(expanded_shape)
            contributions = torch.where(linear > 0, linear * expanded_lower, linear * expanded_upper)
            sum_dims = tuple(range(-input_ndim, 0))
            return contributions.sum(dim=sum_dims) if sum_dims else contributions

        raise ValueError(
            f"linear input axes {tuple(linear_input_axes)} are incompatible with input shape {tuple(input_shape)} "
            f"derived from region shape {tuple(region_shape)}"
        )

    @staticmethod
    @dispatch
    def _maximize_affine_term(region: SimpleRegion, linear: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
        raise NotImplementedError(f"Concretization is not implemented for region type {type(region).__name__}")

    @staticmethod
    @dispatch
    def _maximize_affine_term(region: HyperRectangle, linear: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:  # noqa: F811
        input_lower = region.lower
        input_upper = region.upper

        region_shape = torch.Size(region.shape)
        linear_input_axes = torch.Size(linear.shape[len(output_shape) :])
        batch_shape, input_shape = LinearBounds._split_region_shape(region_shape, output_shape, linear_input_axes)
        input_ndim = len(input_shape)
        output_ndim = len(output_shape)

        if linear_input_axes == input_shape:
            expanded_shape = (*batch_shape, *([1] * (output_ndim - len(batch_shape))), *input_shape)
            expanded_lower = input_lower.reshape(expanded_shape)
            expanded_upper = input_upper.reshape(expanded_shape)
            contributions = torch.where(linear > 0, linear * expanded_upper, linear * expanded_lower)
            sum_dims = tuple(range(-input_ndim, 0))
            return contributions.sum(dim=sum_dims) if sum_dims else contributions

        raise ValueError(
            f"linear input axes {tuple(linear_input_axes)} are incompatible with input shape {tuple(input_shape)} "
            f"derived from region shape {tuple(region_shape)}"
        )

    @staticmethod
    def combine_linear_terms(
        components: list[tuple[LinearBounds, Literal["lower", "upper"], float]],
    ) -> tuple[list[SimpleRegion], list[torch.Tensor], list[int]]:
        merged: dict[int, tuple[SimpleRegion, torch.Tensor]] = {}
        ordered_input_ids: list[int] = []

        for bounds, bound_side, scale in components:
            linears = bounds.linear_lowers if bound_side == "lower" else bounds.linear_uppers
            for input_id, region, linear in zip(bounds.input_ids, bounds.regions, linears, strict=True):
                contribution = linear if scale == 1 else scale * linear
                if input_id in merged:
                    existing_region, existing_linear = merged[input_id]
                    if existing_region.shape != region.shape:
                        raise ValueError(
                            "Cannot merge input_id "
                            f"{input_id}: region shapes differ "
                            f"{existing_region.shape} vs {region.shape}"
                        )
                    merged[input_id] = (existing_region, existing_linear + contribution)
                else:
                    ordered_input_ids.append(input_id)
                    merged[input_id] = (region, contribution)

        regions = [merged[input_id][0] for input_id in ordered_input_ids]
        linear_terms = [merged[input_id][1] for input_id in ordered_input_ids]
        return regions, linear_terms, ordered_input_ids

    @property
    def regions(self) -> list[SimpleRegion]:
        """Get input regions associated with these bounds."""
        return self._regions.copy()

    @property
    def region(self) -> AbstractRegion:
        """Get the single input region associated with these bounds."""
        if len(self._regions) != 1:
            raise ValueError(f"LinearBounds has {len(self._regions)} regions; use regions instead of region")
        return self._regions[0]

    @property
    def input_ids(self) -> list[int]:
        """Get input IDs associated with the affine terms."""
        return self._input_ids.copy()

    @property
    def linear_lowers(self) -> list[torch.Tensor]:
        """Get linear coefficients for the lower bound, one tensor per input region."""
        return self._linear_lower.copy()

    @property
    def linear_lower(self) -> torch.Tensor | None:
        """Get linear coefficients for the lower bound in the single-input case."""
        if not self._linear_lower:
            return None
        if len(self._linear_lower) != 1:
            raise ValueError(
                f"LinearBounds has {len(self._linear_lower)} lower coefficient tensors; use linear_lowers instead"
            )
        return self._linear_lower[0]

    @property
    def bias_lower(self) -> torch.Tensor:
        """Get bias term for lower bound (b_l)."""
        return self._bias_lower

    @property
    def linear_uppers(self) -> list[torch.Tensor]:
        """Get linear coefficients for the upper bound, one tensor per input region."""
        return self._linear_upper.copy()

    @property
    def linear_upper(self) -> torch.Tensor | None:
        """Get linear coefficients for the upper bound in the single-input case."""
        if not self._linear_upper:
            return None
        if len(self._linear_upper) != 1:
            raise ValueError(
                f"LinearBounds has {len(self._linear_upper)} upper coefficient tensors; use linear_uppers instead"
            )
        return self._linear_upper[0]

    @property
    def bias_upper(self) -> torch.Tensor:
        """Get bias term for upper bound (b_u)."""
        return self._bias_upper

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape of bounded tensor."""
        return tuple(self.bias_lower.shape)

    @property
    def input_dim(self) -> int:
        """Get input dimensions of linear bounds."""
        if not self._linear_lower:
            return 0

        return sum(
            torch.Size(
                self._split_region_shape(
                    torch.Size(region.shape),
                    self.bias_lower.shape,
                    torch.Size(linear.shape[len(self.bias_lower.shape) :]),
                )[1]
            ).numel()
            for region, linear in zip(self._regions, self._linear_lower, strict=True)
        )

    @staticmethod
    def _move_region(region: SimpleRegion, device: str | torch.device) -> SimpleRegion:
        moved_region = region.to(device)
        if not isinstance(moved_region, SimpleRegion):
            raise TypeError(f"Expected SimpleRegion after to(...), got {type(moved_region).__name__}")
        return moved_region

    @property
    def device(self) -> torch.device:
        """Get device of bounds."""
        return self.bias_lower.device

    def to(self, device: str | torch.device) -> LinearBounds:
        """Move bounds to a device."""
        return LinearBounds(
            regions=[self._move_region(region, device) for region in self._regions],
            linear_lower=[linear.to(device) for linear in self._linear_lower],
            bias_lower=self.bias_lower.to(device),
            linear_upper=[linear.to(device) for linear in self._linear_upper],
            bias_upper=self.bias_upper.to(device),
            input_ids=self._input_ids.copy() if self._input_ids else None,
        )

    def __getitem__(self, item) -> LinearBounds:
        """Slice/index the bounds."""
        output_item = item if isinstance(item, tuple) else (item,)

        def _index_linear(linear: torch.Tensor, region: SimpleRegion) -> torch.Tensor:
            _, input_shape = self._split_region_shape(
                torch.Size(region.shape),
                self.bias_lower.shape,
                torch.Size(linear.shape[len(self.bias_lower.shape) :]),
            )
            input_ndim = len(input_shape)
            if input_ndim > 0:
                linear_item = output_item + (slice(None),) * input_ndim
            else:
                linear_item = output_item
            return linear[linear_item]

        return LinearBounds(
            regions=self._regions,
            linear_lower=[
                _index_linear(linear, region) for linear, region in zip(self._linear_lower, self._regions, strict=True)
            ],
            bias_lower=self.bias_lower[item],
            linear_upper=[
                _index_linear(linear, region) for linear, region in zip(self._linear_upper, self._regions, strict=True)
            ],
            bias_upper=self.bias_upper[item],
            input_ids=self._input_ids.copy() if self._input_ids else None,
        )

    def concretize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concretize bounds to interval bounds.

        This method uses the input region to convert symbolic/affine bounds
        into concrete interval bounds. The default implementation assumes that
        the bounds are already concrete intervals and simply returns them.

        Subclasses with more complex bound types (e.g., linear bounds) should
        override this method to perform the necessary concretization logic.

        Returns:
            IntervalBounds representing the concretized bounds
        """
        lower_result = self.bias_lower.clone()
        upper_result = self.bias_upper.clone()

        for region, linear_lower in zip(self._regions, self._linear_lower, strict=True):
            lower_result = lower_result + self._minimize_affine_term(region, linear_lower, self.bias_lower.shape)

        for region, linear_upper in zip(self._regions, self._linear_upper, strict=True):
            upper_result = upper_result + self._maximize_affine_term(region, linear_upper, self.bias_upper.shape)

        return lower_result, upper_result

    def clone(self) -> LinearBounds:
        """Create a deep copy."""
        return LinearBounds(
            regions=self._regions,
            linear_lower=[linear.clone() for linear in self._linear_lower],
            bias_lower=self.bias_lower.clone(),
            linear_upper=[linear.clone() for linear in self._linear_upper],
            bias_upper=self.bias_upper.clone(),
            input_ids=self._input_ids.copy() if self._input_ids else None,
        )

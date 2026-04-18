def _merge_backward_bounds(
    bounds_list: list[LinearBounds],
    bias_lower: torch.Tensor,
    bias_upper: torch.Tensor,
) -> LinearBounds:
    """Merge LinearBounds from multiple backward calls by input_id."""
    merged: dict[int, tuple[SimpleRegion, torch.Tensor, torch.Tensor]] = {}
    ordered_ids: list[int] = []

    for bounds in bounds_list:
        for iid, region, wl, wu in zip(
            bounds.input_ids, bounds.regions, bounds.linear_lowers, bounds.linear_uppers, strict=True
        ):
            if iid in merged:
                merged[iid] = (merged[iid][0], merged[iid][1] + wl, merged[iid][2] + wu)
            else:
                ordered_ids.append(iid)
                merged[iid] = (region, wl, wu)

        bias_lower = bias_lower + bounds.bias_lower
        bias_upper = bias_upper + bounds.bias_upper

    regions = [merged[iid][0] for iid in ordered_ids]
    linear_lower = [merged[iid][1] for iid in ordered_ids]
    linear_upper = [merged[iid][2] for iid in ordered_ids]

    return LinearBounds(
        regions=regions,
        linear_lower=linear_lower or None,
        bias_lower=bias_lower,
        linear_upper=linear_upper or None,
        bias_upper=bias_upper,
        input_ids=ordered_ids or None,
    )

import torch
from torch.utils import _pytree as pytree
from typing import Callable, Any


def associative_scan(fn: Callable, elems: Any, reverse: bool = False, axis: int = 0):
    """
    Performs a scan with an associative binary operation, in parallel.
    PyTorch port of jax.lax.associative_scan.

    Args:
        fn: A callable implementing an associative binary operation r = fn(a, b).
        elems: A (possibly nested) pytree of tensors.
        reverse: If True, scan from end to start.
        axis: The axis along which to scan.
    """

    # Flatten the pytree (handles nested dicts/lists of tensors)
    elems_flat, tree_spec = pytree.tree_flatten(elems)

    if not elems_flat:
        raise ValueError("elems must not be empty")

    # Ensure all elements are tensors
    elems_flat = [torch.as_tensor(e) for e in elems_flat]

    # Canonicalize axis
    ndim = elems_flat[0].ndim
    axis = axis % ndim

    # Validate shapes
    num_elems = elems_flat[0].shape[axis]
    if not all(e.shape[axis] == num_elems for e in elems_flat[1:]):
        raise ValueError(
            f"All input arrays must have the same size {num_elems} along axis {axis}."
        )

    # Handle reverse scan
    if reverse:
        elems_flat = [torch.flip(e, [axis]) for e in elems_flat]

    # Helper to combine flattened inputs using the user function
    def combine(a_flat, b_flat):
        a = pytree.tree_unflatten(a_flat, tree_spec)
        b = pytree.tree_unflatten(b_flat, tree_spec)
        c = fn(a, b)
        c_flat, _ = pytree.tree_flatten(c)
        return c_flat

    # Helper to slice tensors generically along an axis
    def slice_at_axis(t, sl):
        # Build a slice tuple: (:, :, sl, :, :) where sl is at `axis`
        idx = [slice(None)] * t.ndim
        idx[axis] = sl
        return t[tuple(idx)]

    # Helper to concatenate tensors along the axis
    def cat_at_axis(tensors):
        return torch.cat(tensors, dim=axis)

    # Recursive scan implementation (Blelloch algorithm)
    def _scan(elems_curr):
        n = elems_curr[0].shape[axis]

        if n < 2:
            return elems_curr

        # 1. Reduce phase: Combine adjacent pairs
        # Left operands: indices 0, 2, 4... (excluding last if odd length)
        # Right operands: indices 1, 3, 5...

        # Equivalent to JAX: slicing.slice_in_dim(elem, 0, -1, stride=2, axis=axis)
        left_ops = [slice_at_axis(e, slice(0, -1, 2)) for e in elems_curr]
        # Equivalent to JAX: slicing.slice_in_dim(elem, 1, None, stride=2, axis=axis)
        right_ops = [slice_at_axis(e, slice(1, None, 2)) for e in elems_curr]

        reduced_elems = combine(left_ops, right_ops)

        # 2. Recursion
        odd_elems = _scan(reduced_elems)

        # 3. Down-sweep phase: Calculate even elements
        # Even elements depend on the computed odd elements and the original inputs.
        if n % 2 == 0:
            # If even length, we combine odd_elems[:-1] with original[2::2]
            # Ops for indices 2, 4, 6...
            e_left = [slice_at_axis(e, slice(0, -1)) for e in odd_elems]
            e_right = [slice_at_axis(e, slice(2, None, 2)) for e in elems_curr]
        else:
            # If odd length, we use all odd_elems with original[2::2]
            e_left = odd_elems
            e_right = [slice_at_axis(e, slice(2, None, 2)) for e in elems_curr]

        even_calc = combine(e_left, e_right)

        # Reconstruct evens: First element is always original first element
        first_elems = [slice_at_axis(e, slice(0, 1)) for e in elems_curr]

        # Even indices result: [Original[0], Calculated[0], Calculated[1]...]
        even_elems = [cat_at_axis([f, calc]) for f, calc in zip(first_elems, even_calc)]

        # 4. Interleave evens and odds to form the final result
        # even_elems are at indices 0, 2, 4...
        # odd_elems are at indices 1, 3, 5...
        return _interleave_lists(even_elems, odd_elems, axis)

    def _interleave_lists(evens, odds, ax):
        # Efficiently interleave two lists of tensors along an axis
        # Optimized for torch.compile using stack/view where possible
        res = []
        for e, o in zip(evens, odds):
            # Check shapes (dimensions other than axis should match)
            n_e = e.shape[ax]
            n_o = o.shape[ax]

            if n_e == n_o:
                # Perfect interleave (Even total length)
                # Stack along new dimension then flatten
                stacked = torch.stack([e, o], dim=ax + 1)
                # Flatten the stacked dimension back into axis
                new_shape = list(e.shape)
                new_shape[ax] = n_e + n_o
                res.append(stacked.reshape(new_shape))
            elif n_e == n_o + 1:
                # Odd total length (one extra even element at the end)
                # Stack the matching parts
                e_part = slice_at_axis(e, slice(0, -1))
                last_e = slice_at_axis(e, slice(-1, None))

                stacked = torch.stack([e_part, o], dim=ax + 1)

                flat_shape = list(e_part.shape)
                flat_shape[ax] = n_e - 1 + n_o
                flattened = stacked.reshape(flat_shape)

                res.append(torch.cat([flattened, last_e], dim=ax))
            else:
                # Fallback for safety (should not be reached in standard scan)
                raise RuntimeError(
                    f"Shape mismatch in interleave: {e.shape} vs {o.shape}"
                )
        return res

    # Run the scan
    scanned_flat = _scan(elems_flat)

    # Un-reverse if needed
    if reverse:
        scanned_flat = [torch.flip(s, [axis]) for s in scanned_flat]

    return pytree.tree_unflatten(scanned_flat, tree_spec)

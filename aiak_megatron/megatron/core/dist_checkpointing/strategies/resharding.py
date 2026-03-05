# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

""" Performant resharding of flattened tensors.

Tensors that are first sharded (e.g. across TP) and then flattened cause
very irregular access patterns during loading. The idea for performant save/load
is to store tensors with global shape [X, Y, Z] and local shape [x, y, z]
as tensors with global shape [X // x, Y // y, Z // z, x * y * z] and
local shape [1, 1, 1, x * y * z]. This allows parallel save of tensors along the
last (flattened) dimension. During loading, some additional resharding is needed.
"""
import logging
import math
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.distributed.checkpoint import ChunkStorageMetadata
from torch.distributed.checkpoint.resharding import _shards_get_overlap_region_wrt_saved_tensor

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.dict_utils import (
    dict_list_map_inplace,
    extract_matching_values,
)
from megatron.core.dist_checkpointing.mapping import (
    ShardedStateDict,
    ShardedTensorFactory,
    StateDict,
    apply_factories,
    apply_factory_merges,
)

logger = logging.getLogger(__name__)


@dataclass
class TensorReformulationMetadata:
    """Metadata needed to restore the original tensor shape.

    Args:
        ckpt_orig_global_shape (Tuple[int, ...]): original global shape of the tensor
            saved in the checkpoint. This is the global shape of the application,
            further reformulated into `ckpt_reform_global_shape` while saving.
        ckpt_reform_global_shape (Tuple[int, ...]): reformulated global shape of the tensor
            saved in the checkpoint. This is the actual saved shape.
    """

    ckpt_orig_global_shape: Tuple[int, ...]
    ckpt_reform_global_shape: Tuple[int, ...]

    def __post_init__(self):
        assert self.ckpt_orig_global_shape


def nd_flattened_tensor_reformulated_global_shape(sh_ten: ShardedTensor) -> Tuple[int, ...]:
    """Reformulated global shape of the flattened N-D ShardedTensor.

    N-D tensor global shape [X, Y, Z] and local shape [x, y, z]
    is reformulated into global shape [X // x, Y // y, Z // z, x * y * z] and
    local shape [1, 1, 1, x * y * z], to allow parallel save of tensors along the
    last (flattened) dimension.

    Args:
        sh_ten (ShardedTensor): flattened N-D ShardedTensor (N > 1)

    Returns:
        Tuple[int, ...]: reformulated tensor shape
    """
    assert is_nd_flattened_tensor(sh_ten), sh_ten
    return sh_ten.axis_fragmentations + (int(np.prod(sh_ten.local_shape)),)


def is_nd_flattened_tensor(sh_ten: Any) -> bool:
    """Checks if ShardedTensor is flattened and more than 1-dimensional

    Args:
        sh_ten (Any): any object

    Returns:
        bool: whether the given object is a flattened ShardedTensor and is N-dimensional (N > 1)
    """
    return isinstance(sh_ten, ShardedTensor) and sh_ten.flattened_range is not None


# information needed to restore. With current implementation, this is a nested state dict
# with ShardedTensorFactories which is basically a ShardedStateDict type
ReformulationRestoreMetadata = ShardedStateDict


def apply_nd_flattened_tensors_reformulation(
    sharded_state_dict: ShardedStateDict,
    reformulation_metadata: Dict[str, TensorReformulationMetadata],
) -> Tuple[ShardedStateDict, ReformulationRestoreMetadata]:
    """Applies N-D reformulation to a given sharded state dict.

    After applying the method and loading the reformulated state dict,
    the `restore_nd_flattened_tensors_formulation` needs to be applied.

    Current implementation uses ShardedTensorFactories for convenience of
    restoring the original structure, but it's just an implementation detail.
    Turns N-D ShardedTensors into factories and immediately applies them,
    keeping the data needed to restore the original structure.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict potentially
            with tensors to reformulate.
        reformulation_metadata (Dict[str, TensorReformulationMetadata]): dict
            containing all metadata needed for reformulating tensors in `sharded_state_dict`.
            for each N-D flattened tensor `sh_ten` in `sharded_state_dict` there must be an
            entry with `sh_ten.key`.

    Returns:
        tuple:
            ShardedStateDict - reformulated sharded state dict
            ReformulationRestoreMetadata - data needed to restore the original formulation
                with `restore_nd_flattened_tensors_formulation`
    """

    def maybe_reformulate_nd_flattened_tensor(sh_ten: Any):
        if not isinstance(sh_ten, ShardedTensor) or not is_nd_flattened_tensor(sh_ten):
            return sh_ten
        # N-D flattened ShardedTensor
        try:
            sh_ten_reformulation_metadata = reformulation_metadata[sh_ten.key]
        except KeyError as e:
            # Handle legacy checkpointing where 1-D flatten tensor metadata was not saved
            if len(sh_ten.global_shape) == 1:
                return sh_ten
            raise CheckpointingException(
                f'Missing reformulation metadata for tensor {sh_ten}. '
                f'Existing keys: {reformulation_metadata.keys()}'
            ) from e

        ckpt_actual_saved_shape = sh_ten_reformulation_metadata.ckpt_reform_global_shape
        app_actual_load_shape = nd_flattened_tensor_reformulated_global_shape(sh_ten)
        if ckpt_actual_saved_shape == app_actual_load_shape:
            # Same shape - no need to reshard
            return sh_ten

        return reformulate_single_nd_flattened_tensor(sh_ten, sh_ten_reformulation_metadata)

    # Turn N-D tensors into factories and immediately apply them
    dict_list_map_inplace(maybe_reformulate_nd_flattened_tensor, sharded_state_dict)
    sh_ten_factories, _ = extract_matching_values(
        sharded_state_dict,
        lambda x: isinstance(x, ShardedTensorFactory),
        return_lists_as_dicts=True,
    )
    apply_factories(sharded_state_dict)

    # Unlink `data` pointers to free memory
    def unlink_data(x):
        x.data = None
        return x

    dict_list_map_inplace(unlink_data, sh_ten_factories)
    return sharded_state_dict, sh_ten_factories


def restore_nd_flattened_tensors_formulation(
    state_dict: StateDict, formulation_restore_metadata: ReformulationRestoreMetadata
) -> StateDict:
    """Restores the original state dict from a reformulated form.

    Inverse of `apply_nd_flattened_tensors_reformulation`.

    Args:
        state_dict (StateDict): state dict obtained by loading a reformulated
            sharded state dict.
        formulation_restore_metadata (ReformulationRestoreMetadata): metadata returned by
            `apply_nd_flattened_tensors_reformulation` function

    Returns:
        StateDict: state dict with the original tensors formulation restored
    """
    return apply_factory_merges(state_dict, formulation_restore_metadata)


def reformulate_single_nd_flattened_tensor(
    sh_ten: ShardedTensor, reformulation_metadata: TensorReformulationMetadata
) -> Union[Any, ShardedTensorFactory]:
    """Reformulates shapes of a single N-D flattened ShardedTensor.

    We need to define a pair of transformations:
    - turn N-D ShardedTensor with original formulation into multiple reformulated ShardedTensors
    - merge multiple reformulated loaded torch.Tensors into a single original tensor
    Current implementation uses ShardedTensorFactories as a convenient mechanism
    for specifying and keeping track of those transformations.

    Args:
        sh_ten (ShardedTensor): sharded tensor to reformulate.
        reformulation_metadata (TensorReformulationMetadata): metadata needed to
            perform the reformulation

    Returns:
        ShardedTensorFactory: factory that keeps information how to reformulate
            (build) the ShardedTensor and then restore original formulation (merge)
            after loading.
    """
    rmd = reformulation_metadata
    # Data won't be needed - remove unnecessary tensor references
    sh_ten = sh_ten.without_data()

    # Based on reformulation_metadata, determine other tensor shapes and metadata
    ckpt_axis_fragmentation = rmd.ckpt_reform_global_shape[:-1]
    for sh, fragm in zip(rmd.ckpt_orig_global_shape, ckpt_axis_fragmentation):
        assert sh % fragm == 0, (sh_ten, rmd.ckpt_reform_global_shape)
    ckpt_local_shape_with_prepended_axis = tuple(
        sh // fragm for sh, fragm in zip(rmd.ckpt_orig_global_shape, ckpt_axis_fragmentation)
    )
    assert (
        ckpt_local_shape_with_prepended_axis[: sh_ten.prepend_axis_num]
        == (1,) * sh_ten.prepend_axis_num
    ), (ckpt_local_shape_with_prepended_axis, sh_ten)
    ckpt_local_shape = ckpt_local_shape_with_prepended_axis[sh_ten.prepend_axis_num :]

    # Iterate over reformulated shapes needed by the application and from checkpoint,
    # and generate new ShardedTensors that match the checkpoint sharding.
    overlap_dim_offsets = []
    assert len(ckpt_axis_fragmentation) == len(sh_ten.axis_fragmentations), (
        ckpt_axis_fragmentation,
        sh_ten,
    )
    for dim, (app_chunk_dim_offset, ckpt_fragm, app_fragm) in enumerate(
        zip(
            sh_ten.local_chunk_offset_in_global(),
            ckpt_axis_fragmentation,
            sh_ten.axis_fragmentations,
        )
    ):
        # without `int`, it's an exact offset of the app shard expressed in ckpt_local_shape units
        first_overlap_dim_offset = int(ckpt_fragm / app_fragm * app_chunk_dim_offset)
        # `math.ceil` argument is an exact offset of the app next shard expressed
        # in ckpt_local_shape units
        next_overlap_dim_offset = math.ceil(ckpt_fragm / app_fragm * (app_chunk_dim_offset + 1))
        overlap_dim_offsets.append(range(first_overlap_dim_offset, next_overlap_dim_offset))

    logger.debug(
        f'Generated the following number of overlap shards for each dimension: '
        f'{list(map(len, overlap_dim_offsets))} for fragmentation ckpt '
        f'{ckpt_axis_fragmentation} vs app {sh_ten.axis_fragmentations} '
        f'and chunk offset {sh_ten.local_chunk_offset_in_global()}'
    )
    reformulated_sh_tens = {}
    for chunk_offset in product(*overlap_dim_offsets):
        global_offset = tuple(
            chunk_off * chunk_shape
            for chunk_off, chunk_shape in zip(chunk_offset, ckpt_local_shape_with_prepended_axis)
        )
        reformulated_sh_tens[(global_offset, ckpt_local_shape)] = ShardedTensor(
            sh_ten.key,
            None,
            sh_ten.dtype,
            ckpt_local_shape,
            rmd.ckpt_orig_global_shape,
            global_offset,
            ckpt_axis_fragmentation,
            sh_ten.replica_id,
            sh_ten.prepend_axis_num,
            sh_ten.allow_shape_mismatch,
            flattened_range=slice(0, rmd.ckpt_reform_global_shape[-1]),  # whole ckpt shard
        )

    # Now, we have to define the transformations from application sharding
    # to checkpoint sharding.

    @torch.no_grad()
    def sh_ten_build_fn(*args, **kwargs):
        # Here we simply return the precomputed tensors.
        return reformulated_sh_tens

    @torch.no_grad()
    def sh_ten_merge_fn(sub_state_dict):
        # This is the non-flattened local tensor with original formulation
        # that we are going to fill with shards loaded from the checkpoint.
        app_non_flat_ten = torch.empty(
            sh_ten.local_shape,
            dtype=sh_ten.dtype,
            device=sh_ten.data.device if sh_ten.data is not None else None,
        )

        assert len(sub_state_dict) > 0
        for (ckpt_global_offset, ckpt_local_shape), ckpt_ten in sub_state_dict.items():
            # For each ckpt shard, we fill the appropriate application shard part
            dest_ten = app_non_flat_ten
            src_ten = ckpt_ten.view(ckpt_local_shape)
            # We don't need narrowing over `prepend_axis_num` axes so we take
            # the [sh_ten.prepend_axis_num:] offsets slice
            for (
                dim,
                offset_for_saved_tensor,
                offset_for_current_tensor,
                length,
            ) in _shards_get_overlap_region_wrt_saved_tensor(
                saved_shard=ChunkStorageMetadata(
                    ckpt_global_offset[sh_ten.prepend_axis_num :], ckpt_local_shape
                ),
                current_shard=ChunkStorageMetadata(
                    sh_ten.global_offset[sh_ten.prepend_axis_num :], sh_ten.local_shape
                ),
            ):
                src_ten = src_ten.narrow(dim, offset_for_saved_tensor, length)
                dest_ten = dest_ten.narrow(dim, offset_for_current_tensor, length)
            dest_ten.copy_(src_ten)
        return app_non_flat_ten.flatten()[sh_ten.flattened_range]

    return ShardedTensorFactory(
        sh_ten.key,
        sh_ten.data,
        sh_ten_build_fn,
        sh_ten_merge_fn,
        sh_ten.replica_id,
        sh_ten.flattened_range,
    )


# ---- Non-flat (regular model weight) tensor resharding for TP/PP changes ----

def _get_axis_fragmentations_from_storage_data(
    key: str,
    global_shape: Tuple[int, ...],
    metadata: Any,
) -> Optional[Tuple[int, ...]]:
    """Detect checkpoint's axis_fragmentations from storage_data.

    For uniformly sharded tensors, the unique offsets per axis correspond to
    the axis_fragmentations used when saving.

    Args:
        key (str): tensor key
        global_shape (Tuple[int, ...]): tensor global shape
        metadata: PyT checkpoint Metadata object

    Returns:
        Tuple of axis fragmentations inferred from stored chunks, or None if
        storage_data is unavailable or the key is not found.
    """
    storage_data = getattr(metadata, 'storage_data', None)
    if storage_data is None:
        return None

    unique_offsets_per_axis = [set() for _ in global_shape]
    found_key = False
    for mi in storage_data.keys():
        if getattr(mi, 'fqn', None) == key:
            offset = getattr(mi, 'offset', None)
            if offset is not None and len(offset) == len(global_shape):
                found_key = True
                for dim, off in enumerate(offset):
                    unique_offsets_per_axis[dim].add(int(off))

    if not found_key:
        return None

    return tuple(len(offsets) if offsets else 1 for offsets in unique_offsets_per_axis)


def get_non_flat_resharding_metadata(
    sharded_state_dict: ShardedStateDict,
    checkpoint_metadata: Any,
) -> Dict[str, Tuple[int, ...]]:
    """Get resharding metadata for non-flat ShardedTensors that need TP/PP resharding.

    Compares the checkpoint's axis_fragmentations (from mcore_data or storage_data)
    with the current model's axis_fragmentations. Returns metadata for tensors
    that need resharding.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict with ShardedTensors
        checkpoint_metadata: PyT Metadata from the checkpoint

    Returns:
        Dict mapping tensor keys to checkpoint axis_fragmentations for tensors
        that need resharding.
    """
    resharding_needed: Dict[str, Tuple[int, ...]] = {}
    mcore_data = getattr(checkpoint_metadata, 'mcore_data', {}) or {}

    seen_keys: set = set()
    from megatron.core.dist_checkpointing.dict_utils import nested_values
    for sh_ten in nested_values(sharded_state_dict):
        if not isinstance(sh_ten, ShardedTensor) or is_nd_flattened_tensor(sh_ten):
            continue
        if sh_ten.key in seen_keys:
            continue
        if sh_ten.allow_shape_mismatch:
            continue
        seen_keys.add(sh_ten.key)

        if sh_ten.key not in checkpoint_metadata.state_dict_metadata:
            continue

        # Try to get checkpoint's axis_fragmentations from mcore_data first
        ckpt_mcore = mcore_data.get(sh_ten.key, {})
        ckpt_axis_fragmentations = ckpt_mcore.get('saved_axis_fragmentations')

        if ckpt_axis_fragmentations is None:
            # Old checkpoint: try to detect from storage_data
            ckpt_axis_fragmentations = _get_axis_fragmentations_from_storage_data(
                sh_ten.key, sh_ten.global_shape, checkpoint_metadata
            )

        if ckpt_axis_fragmentations is None:
            # Cannot determine checkpoint's sharding; skip
            continue

        ckpt_axis_fragmentations = tuple(ckpt_axis_fragmentations)

        # No resharding needed if the axis_fragmentations match
        if ckpt_axis_fragmentations == tuple(sh_ten.axis_fragmentations):
            continue

        resharding_needed[sh_ten.key] = ckpt_axis_fragmentations

    return resharding_needed


def apply_non_flat_tensors_resharding(
    sharded_state_dict: ShardedStateDict,
    resharding_metadata: Dict[str, Tuple[int, ...]],
) -> Tuple[ShardedStateDict, ReformulationRestoreMetadata]:
    """Apply resharding for non-flat ShardedTensors with TP/PP change.

    Analogous to apply_nd_flattened_tensors_reformulation but for non-flat
    (regular model weight) tensors.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict potentially
            with non-flat tensors to reshard.
        resharding_metadata (Dict[str, Tuple[int, ...]]): dict mapping tensor keys
            to checkpoint axis_fragmentations for tensors that need resharding.

    Returns:
        tuple:
            ShardedStateDict - resharded sharded state dict
            ReformulationRestoreMetadata - data needed to restore the original
                formulation with restore_non_flat_tensors_resharding
    """

    def maybe_reformulate_non_flat_tensor(sh_ten: Any):
        if not isinstance(sh_ten, ShardedTensor) or is_nd_flattened_tensor(sh_ten):
            return sh_ten
        if sh_ten.key not in resharding_metadata:
            return sh_ten
        return reformulate_single_non_flat_tensor(sh_ten, resharding_metadata[sh_ten.key])

    dict_list_map_inplace(maybe_reformulate_non_flat_tensor, sharded_state_dict)
    sh_ten_factories, _ = extract_matching_values(
        sharded_state_dict,
        lambda x: isinstance(x, ShardedTensorFactory),
        return_lists_as_dicts=True,
    )
    apply_factories(sharded_state_dict)

    def unlink_data(x):
        x.data = None
        return x

    dict_list_map_inplace(unlink_data, sh_ten_factories)
    return sharded_state_dict, sh_ten_factories


def restore_non_flat_tensors_resharding(
    state_dict: StateDict,
    formulation_restore_metadata: ReformulationRestoreMetadata,
) -> StateDict:
    """Restores the original state dict from a resharded non-flat form.

    Inverse of apply_non_flat_tensors_resharding.

    Args:
        state_dict (StateDict): state dict obtained by loading a resharded
            sharded state dict.
        formulation_restore_metadata (ReformulationRestoreMetadata): metadata
            returned by apply_non_flat_tensors_resharding.

    Returns:
        StateDict: state dict with the original tensors formulation restored
    """
    return apply_factory_merges(state_dict, formulation_restore_metadata)


def reformulate_single_non_flat_tensor(
    sh_ten: ShardedTensor,
    ckpt_axis_fragmentations: Tuple[int, ...],
) -> ShardedTensorFactory:
    """Reformulate a single non-flat ShardedTensor for loading with TP/PP change.

    Creates a ShardedTensorFactory that:
    - build_fn: produces ShardedTensors matching the checkpoint's sharding
    - merge_fn: slices the loaded checkpoint shard to the current rank's portion

    Args:
        sh_ten (ShardedTensor): non-flat sharded tensor to reformulate.
        ckpt_axis_fragmentations (Tuple[int, ...]): axis_fragmentations from the
            checkpoint (the sharding used when saving).

    Returns:
        ShardedTensorFactory: factory keeping reformulation and merge information.
    """
    # Data won't be needed during loading
    sh_ten = sh_ten.without_data()

    # Compute the checkpoint's local shape = global_shape / ckpt_axis_fragmentations
    assert len(ckpt_axis_fragmentations) == len(sh_ten.global_shape), (
        ckpt_axis_fragmentations, sh_ten
    )
    for sh, fragm in zip(sh_ten.global_shape, ckpt_axis_fragmentations):
        assert sh % fragm == 0, (sh_ten, ckpt_axis_fragmentations)
    ckpt_local_shape_with_prepended = tuple(
        sh // fragm for sh, fragm in zip(sh_ten.global_shape, ckpt_axis_fragmentations)
    )
    ckpt_local_shape = ckpt_local_shape_with_prepended[sh_ten.prepend_axis_num :]

    # Determine which checkpoint chunks overlap with the current shard
    overlap_dim_offsets = []
    assert len(ckpt_axis_fragmentations) == len(sh_ten.axis_fragmentations), (
        ckpt_axis_fragmentations, sh_ten
    )
    for dim, (app_chunk_dim_offset, ckpt_fragm, app_fragm) in enumerate(
        zip(
            sh_ten.local_chunk_offset_in_global(),
            ckpt_axis_fragmentations,
            sh_ten.axis_fragmentations,
        )
    ):
        first_overlap_dim_offset = int(ckpt_fragm / app_fragm * app_chunk_dim_offset)
        next_overlap_dim_offset = math.ceil(ckpt_fragm / app_fragm * (app_chunk_dim_offset + 1))
        overlap_dim_offsets.append(range(first_overlap_dim_offset, next_overlap_dim_offset))

    logger.debug(
        f'Non-flat resharding: generated overlap shards per dimension: '
        f'{list(map(len, overlap_dim_offsets))} for ckpt fragmentation '
        f'{ckpt_axis_fragmentations} vs app {sh_ten.axis_fragmentations} '
        f'and chunk offset {sh_ten.local_chunk_offset_in_global()}'
    )

    reformulated_sh_tens = {}
    for chunk_offset in product(*overlap_dim_offsets):
        global_offset = tuple(
            chunk_off * chunk_shape
            for chunk_off, chunk_shape in zip(chunk_offset, ckpt_local_shape_with_prepended)
        )
        reformulated_sh_tens[(global_offset, ckpt_local_shape)] = ShardedTensor(
            sh_ten.key,
            None,
            sh_ten.dtype,
            ckpt_local_shape,
            sh_ten.global_shape,
            global_offset,
            ckpt_axis_fragmentations,
            sh_ten.replica_id,
            sh_ten.prepend_axis_num,
            sh_ten.allow_shape_mismatch,
            # NO flattened_range for non-flat tensors
        )

    @torch.no_grad()
    def sh_ten_build_fn(*args, **kwargs):
        return reformulated_sh_tens

    @torch.no_grad()
    def sh_ten_merge_fn(sub_state_dict):
        assert len(sub_state_dict) > 0
        # Determine device from loaded tensors
        first_val = next(iter(sub_state_dict.values()))
        device = first_val.device if isinstance(first_val, torch.Tensor) else None
        app_ten = torch.empty(sh_ten.local_shape, dtype=sh_ten.dtype, device=device)

        for (ckpt_global_offset, ckpt_local_shape_key), ckpt_ten in sub_state_dict.items():
            dest_ten = app_ten
            src_ten = ckpt_ten.view(ckpt_local_shape_key)
            for (
                dim,
                offset_for_saved_tensor,
                offset_for_current_tensor,
                length,
            ) in _shards_get_overlap_region_wrt_saved_tensor(
                saved_shard=ChunkStorageMetadata(
                    ckpt_global_offset[sh_ten.prepend_axis_num :], ckpt_local_shape_key
                ),
                current_shard=ChunkStorageMetadata(
                    sh_ten.global_offset[sh_ten.prepend_axis_num :], sh_ten.local_shape
                ),
            ):
                src_ten = src_ten.narrow(dim, offset_for_saved_tensor, length)
                dest_ten = dest_ten.narrow(dim, offset_for_current_tensor, length)
            dest_ten.copy_(src_ten)

        return app_ten

    return ShardedTensorFactory(
        sh_ten.key,
        sh_ten.data,
        sh_ten_build_fn,
        sh_ten_merge_fn,
        sh_ten.replica_id,
        # NO flattened_range for non-flat tensors
    )

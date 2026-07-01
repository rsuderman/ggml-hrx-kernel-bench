from __future__ import annotations

from ggml_hrx_kernel_bench.routing.v2.descriptors import (
    ConcreteTensor,
    ConcreteTensorDimension,
    DimensionBounds,
    RouteConstraints,
    StrideDescriptor,
    TensorDescriptor,
    TensorDimensionDescriptor,
    TensorStrideIdentifier,
    tensor_accepts_descriptor,
)


def test_tensor_descriptor_matches_contiguous_row_major_tensor() -> None:
    descriptor = TensorDescriptor(
        dtype="F32",
        dimensions=(
            TensorDimensionDescriptor(name="ncols"),
            TensorDimensionDescriptor(name="nrows"),
        ),
        strides=(
            TensorStrideIdentifier(name="stride_cols"),
            TensorStrideIdentifier(name="stride_rows"),
        ),
    )
    constraints = RouteConstraints(
        sizes={
            "ncols": DimensionBounds(min=1, max=1024),
            "nrows": DimensionBounds(min=1, max=1024),
        },
        strides={
            "stride_cols": StrideDescriptor(value=1),
            "stride_rows": StrideDescriptor(dimension="ncols"),
        },
    )
    tensor = ConcreteTensor(
        dtype="F32",
        dimensions=(
            ConcreteTensorDimension(name="ncols", size=64, stride=1),
            ConcreteTensorDimension(name="nrows", size=128, stride=64),
        ),
    )

    assert tensor_accepts_descriptor(descriptor, constraints, tensor) is True


def test_tensor_descriptor_rejects_transposed_tensor_order() -> None:
    descriptor = TensorDescriptor(
        dtype="F32",
        dimensions=(
            TensorDimensionDescriptor(name="ncols"),
            TensorDimensionDescriptor(name="nrows"),
        ),
        strides=(
            TensorStrideIdentifier(name="stride_cols"),
            TensorStrideIdentifier(name="stride_rows"),
        ),
    )
    constraints = RouteConstraints(
        sizes={
            "ncols": DimensionBounds(min=1, max=1024),
            "nrows": DimensionBounds(min=1, max=1024),
        },
        strides={
            "stride_cols": StrideDescriptor(value=1),
            "stride_rows": StrideDescriptor(dimension="ncols"),
        },
    )
    tensor = ConcreteTensor(
        dtype="F32",
        dimensions=(
            ConcreteTensorDimension(name="nrows", size=128, stride=1),
            ConcreteTensorDimension(name="ncols", size=64, stride=128),
        ),
    )

    assert tensor_accepts_descriptor(descriptor, constraints, tensor) is False


def test_tensor_descriptor_rejects_stride_mismatch() -> None:
    descriptor = TensorDescriptor(
        dtype="F32",
        dimensions=(
            TensorDimensionDescriptor(name="ncols"),
            TensorDimensionDescriptor(name="nrows"),
        ),
        strides=(
            TensorStrideIdentifier(name="stride_cols"),
            TensorStrideIdentifier(name="stride_rows"),
        ),
    )
    constraints = RouteConstraints(
        sizes={
            "ncols": DimensionBounds(min=1, max=1024),
            "nrows": DimensionBounds(min=1, max=1024),
        },
        strides={
            "stride_cols": StrideDescriptor(value=1),
            "stride_rows": StrideDescriptor(dimension="ncols"),
        },
    )
    tensor = ConcreteTensor(
        dtype="F32",
        dimensions=(
            ConcreteTensorDimension(name="ncols", size=64, stride=1),
            ConcreteTensorDimension(name="nrows", size=128, stride=32),
        ),
    )

    assert tensor_accepts_descriptor(descriptor, constraints, tensor) is False

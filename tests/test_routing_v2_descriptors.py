from __future__ import annotations

from ggml_hrx_kernel_bench.routing.v2.matching import route_accepts_tensors, tensor_accepts_descriptor
from ggml_hrx_kernel_bench.routing.v2.models import (
    ConcreteTensor,
    ConcreteTensorDimension,
    ConstraintCheck,
    RouteConstraints,
    TensorDescriptor,
    V2Route,
    ValueDefinition,
)


def _contiguous_constraint_setup() -> tuple[TensorDescriptor, RouteConstraints, tuple[ValueDefinition, ...]]:
    descriptor = TensorDescriptor(
        dtype="F32",
        dimensions_capture="dimensions",
        strides_capture="strides",
    )
    constraints = RouteConstraints(
        checks=(
            ConstraintCheck(equals=("contiguous_strides", "strides")),
        )
    )
    values = (
        ValueDefinition(name="contiguous_strides", contiguous_strides="dimensions"),
        ValueDefinition(name="total_size", product="dimensions"),
    )
    return descriptor, constraints, values


def test_tensor_descriptor_matches_contiguous_row_major_tensor() -> None:
    descriptor, constraints, values = _contiguous_constraint_setup()
    tensor = ConcreteTensor(
        dtype="F32",
        dimensions=(
            ConcreteTensorDimension(name="ncols", size=64, stride=1),
            ConcreteTensorDimension(name="nrows", size=128, stride=64),
        ),
    )

    assert tensor_accepts_descriptor(
        descriptor,
        constraints,
        tensor,
        computed_values=values,
    ) is True


def test_tensor_descriptor_accepts_higher_rank_contiguous_tensor() -> None:
    descriptor, constraints, values = _contiguous_constraint_setup()
    tensor = ConcreteTensor(
        dtype="F32",
        dimensions=(
            ConcreteTensorDimension(name="d0", size=8, stride=1),
            ConcreteTensorDimension(name="d1", size=8, stride=8),
            ConcreteTensorDimension(name="d2", size=4, stride=64),
        ),
    )

    assert tensor_accepts_descriptor(
        descriptor,
        constraints,
        tensor,
        computed_values=values,
    ) is True


def test_tensor_descriptor_accepts_small_total_size_when_strides_match() -> None:
    descriptor, constraints, values = _contiguous_constraint_setup()
    tensor = ConcreteTensor(
        dtype="F32",
        dimensions=(
            ConcreteTensorDimension(name="d0", size=10, stride=1),
            ConcreteTensorDimension(name="d1", size=10, stride=10),
        ),
    )

    assert tensor_accepts_descriptor(
        descriptor,
        constraints,
        tensor,
        computed_values=values,
    ) is True


def test_tensor_descriptor_rejects_stride_mismatch() -> None:
    descriptor, constraints, values = _contiguous_constraint_setup()
    tensor = ConcreteTensor(
        dtype="F32",
        dimensions=(
            ConcreteTensorDimension(name="ncols", size=64, stride=1),
            ConcreteTensorDimension(name="nrows", size=128, stride=32),
        ),
    )

    assert tensor_accepts_descriptor(
        descriptor,
        constraints,
        tensor,
        computed_values=values,
    ) is False


def test_tensor_descriptor_accepts_iota_permutation_capture() -> None:
    descriptor = TensorDescriptor(
        dtype="F32",
        dimensions_capture="dimensions",
        strides_capture="strides",
        permutation_capture="permutation",
    )
    constraints = RouteConstraints(checks=(ConstraintCheck(name="permutation", iota=True),))
    tensor = ConcreteTensor(
        dtype="F32",
        dimensions=(
            ConcreteTensorDimension(name="d0", size=8, stride=1),
            ConcreteTensorDimension(name="d1", size=8, stride=8),
        ),
        permutation=(0, 1),
    )

    assert tensor_accepts_descriptor(
        descriptor,
        constraints,
        tensor,
    ) is True


def test_tensor_descriptor_rejects_non_iota_permutation_capture() -> None:
    descriptor = TensorDescriptor(
        dtype="F32",
        dimensions_capture="dimensions",
        strides_capture="strides",
        permutation_capture="permutation",
    )
    constraints = RouteConstraints(checks=(ConstraintCheck(name="permutation", iota=True),))
    tensor = ConcreteTensor(
        dtype="F32",
        dimensions=(
            ConcreteTensorDimension(name="d0", size=8, stride=1),
            ConcreteTensorDimension(name="d1", size=8, stride=8),
        ),
        permutation=(1, 0),
    )

    assert tensor_accepts_descriptor(
        descriptor,
        constraints,
        tensor,
    ) is False


def test_route_accepts_tensors_requires_equal_dimension_lists() -> None:
    route = V2Route(
        id="add_f32_contiguous_1d",
        family="add_f32",
        op="ADD",
        source_id="add_f32",
        kernel_path="add/contiguous_1d.loom",
        root_symbol="@hrx2_add_f32_contiguous_1d",
        export_name="hrx2_add_f32_contiguous_1d",
        tensors={
            "src0": TensorDescriptor(dtype="F32", dimensions_capture="src0_dimensions", strides_capture="src0_strides"),
            "src1": TensorDescriptor(dtype="F32", dimensions_capture="src1_dimensions", strides_capture="src1_strides"),
            "dst": TensorDescriptor(dtype="F32", dimensions_capture="dst_dimensions", strides_capture="dst_strides"),
        },
        values=(
            ValueDefinition(name="contiguous_strides", contiguous_strides="dst_dimensions"),
            ValueDefinition(name="total_size", product="dst_dimensions"),
        ),
        constraints=RouteConstraints(
            checks=(
                ConstraintCheck(equals=("src0_dimensions", "src1_dimensions", "dst_dimensions")),
                ConstraintCheck(equals=("contiguous_strides", "src0_strides", "src1_strides", "dst_strides")),
            )
        ),
        launch={"workgroup_size": [256, 1, 1]},
        bindings=(),
    )
    tensors = {
        "src0": ConcreteTensor(
            dtype="F32",
            dimensions=(
                ConcreteTensorDimension(name="ncols", size=64, stride=1),
                ConcreteTensorDimension(name="nrows", size=128, stride=64),
            ),
        ),
        "src1": ConcreteTensor(
            dtype="F32",
            dimensions=(
                ConcreteTensorDimension(name="ncols", size=32, stride=1),
                ConcreteTensorDimension(name="nrows", size=128, stride=32),
            ),
        ),
        "dst": ConcreteTensor(
            dtype="F32",
            dimensions=(
                ConcreteTensorDimension(name="ncols", size=64, stride=1),
                ConcreteTensorDimension(name="nrows", size=128, stride=64),
            ),
        ),
    }

    assert route_accepts_tensors(route, tensors) is False


def test_generic_4d_route_accepts_non_contiguous_broadcast_and_repeat_tensors() -> None:
    route = V2Route(
        id="add_f32_generic_4d",
        family="add_f32",
        op="ADD",
        source_id="add_f32",
        kernel_path="add/generic_4d.loom",
        root_symbol="@hrx2_add_f32_generic_4d",
        export_name="hrx2_add_f32_generic_4d",
        tensors={
            "src0": TensorDescriptor(
                dtype="F32",
                dimensions_capture="src0_dimensions",
                strides_capture="src0_strides",
            ),
            "src1": TensorDescriptor(
                dtype="F32",
                dimensions_capture="src1_dimensions",
                strides_capture="src1_strides",
            ),
            "dst": TensorDescriptor(
                dtype="F32",
                dimensions_capture="dst_dimensions",
                strides_capture="dst_strides",
            ),
        },
        values=(
            ValueDefinition(name="total_size", product="dst_dimensions"),
        ),
        constraints=RouteConstraints(
            checks=(
                ConstraintCheck(name="dst_dimensions", length=4),
                ConstraintCheck(divides=("src0_dimensions", "dst_dimensions")),
                ConstraintCheck(divides=("src1_dimensions", "dst_dimensions")),
            )
        ),
        launch={"workgroup_size": [256, 1, 1]},
        bindings=(),
    )
    tensors = {
        "src0": ConcreteTensor(
            dtype="F32",
            dimensions=(
                ConcreteTensorDimension(name="d0", size=4, stride=3),
                ConcreteTensorDimension(name="d1", size=5, stride=29),
                ConcreteTensorDimension(name="d2", size=6, stride=211),
                ConcreteTensorDimension(name="d3", size=7, stride=1703),
            ),
            permutation=(0, 1, 2, 3),
        ),
        "src1": ConcreteTensor(
            dtype="F32",
            dimensions=(
                ConcreteTensorDimension(name="d0", size=2, stride=1),
                ConcreteTensorDimension(name="d1", size=5, stride=4),
                ConcreteTensorDimension(name="d2", size=3, stride=10),
                ConcreteTensorDimension(name="d3", size=7, stride=30),
            ),
            permutation=(1, 2, 0, 3),
        ),
        "dst": ConcreteTensor(
            dtype="F32",
            dimensions=(
                ConcreteTensorDimension(name="d0", size=4, stride=11),
                ConcreteTensorDimension(name="d1", size=5, stride=47),
                ConcreteTensorDimension(name="d2", size=6, stride=263),
                ConcreteTensorDimension(name="d3", size=7, stride=1499),
            ),
            permutation=(0, 1, 2, 3),
        ),
    }

    assert route_accepts_tensors(route, tensors) is True


def test_generic_4d_route_accepts_transposed_src0_and_dst() -> None:
    route = V2Route(
        id="add_f32_generic_4d",
        family="add_f32",
        op="ADD",
        source_id="add_f32",
        kernel_path="add/generic_4d.loom",
        root_symbol="@hrx2_add_f32_generic_4d",
        export_name="hrx2_add_f32_generic_4d",
        tensors={
            "src0": TensorDescriptor(
                dtype="F32",
                dimensions_capture="src0_dimensions",
                strides_capture="src0_strides",
            ),
            "src1": TensorDescriptor(
                dtype="F32",
                dimensions_capture="src1_dimensions",
                strides_capture="src1_strides",
            ),
            "dst": TensorDescriptor(
                dtype="F32",
                dimensions_capture="dst_dimensions",
                strides_capture="dst_strides",
            ),
        },
        values=(ValueDefinition(name="total_size", product="dst_dimensions"),),
        constraints=RouteConstraints(
            checks=(
                ConstraintCheck(name="dst_dimensions", length=4),
                ConstraintCheck(divides=("src0_dimensions", "dst_dimensions")),
                ConstraintCheck(divides=("src1_dimensions", "dst_dimensions")),
            )
        ),
        launch={"workgroup_size": [256, 1, 1]},
        bindings=(),
    )
    tensors = {
        "src0": ConcreteTensor(
            dtype="F32",
            dimensions=(
                ConcreteTensorDimension(name="d0", size=4, stride=3),
                ConcreteTensorDimension(name="d1", size=5, stride=29),
                ConcreteTensorDimension(name="d2", size=6, stride=211),
                ConcreteTensorDimension(name="d3", size=7, stride=1703),
            ),
            permutation=(1, 0, 2, 3),
        ),
        "src1": ConcreteTensor(
            dtype="F32",
            dimensions=(
                ConcreteTensorDimension(name="d0", size=2, stride=1),
                ConcreteTensorDimension(name="d1", size=5, stride=4),
                ConcreteTensorDimension(name="d2", size=3, stride=10),
                ConcreteTensorDimension(name="d3", size=7, stride=30),
            ),
            permutation=(1, 2, 0, 3),
        ),
        "dst": ConcreteTensor(
            dtype="F32",
            dimensions=(
                ConcreteTensorDimension(name="d0", size=4, stride=11),
                ConcreteTensorDimension(name="d1", size=5, stride=47),
                ConcreteTensorDimension(name="d2", size=6, stride=263),
                ConcreteTensorDimension(name="d3", size=7, stride=1499),
            ),
            permutation=(0, 2, 1, 3),
        ),
    }

    assert route_accepts_tensors(route, tensors) is True

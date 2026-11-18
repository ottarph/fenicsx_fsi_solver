# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Compare sampled det(F) with Gmsh element-map bounds without solving an FSI problem.

Run on one process, with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1, for example::

    python -m xfsi_solver.scripts.benchmark_jacobian_range data/meshes/fsi3d/mesh_sec_coarse.xdmf

The two methods compute different quantities on curved reference cells. The output
includes a lower bound on min det(F), obtained from separate deformed/rest bounds,
and validates node ordering against DOLFINx sampling on a subset of cells. Gmsh's
double-precision bounds are subject to its internal tolerance and subdivision limit.
Initialization, form compilation, and one-time reference bounds are excluded from
timings. The bulk timings include displacement packing and rebuilding Gmsh's mesh.
"""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import basix
import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import dolfinx.io.gmsh
import gmsh
import numpy as np
import ufl
from mpi4py import MPI

from xfsi_solver.tools.qoi import deformation_jacobian_range


def timed(function, repeats):
    """Warm up once, then return median and range of repeated wall times in ms."""
    function()
    times = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        times.append(1000 * (perf_counter() - start))
    return {"median_ms": float(np.median(times)), "min_ms": min(times), "max_ms": max(times)}


def benchmark(path, amplitudes, repeats):
    with dolfinx.io.XDMFFile(MPI.COMM_SELF, str(path), "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")
    if mesh.topology.dim != 3 or mesh.geometry.dim != 3:
        raise ValueError("This benchmark requires a three-dimensional volume mesh")
    space = dolfinx.fem.functionspace(mesh, basix.ufl.element("P", mesh.basix_cell(), 2, shape=(3,)))
    displacement = dolfinx.fem.Function(space)
    coordinates = space.tabulate_dof_coordinates()
    cell_dofs = space.dofmap.list
    permutation = dolfinx.io.gmsh.cell_perm_array(mesh.topology.cell_type, cell_dofs.shape[1])
    connectivity = np.asarray(cell_dofs[:, np.argsort(permutation)] + 1, dtype=np.uint64)
    num_cells = mesh.topology.index_map(3).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    element_tags = np.arange(1, num_cells + 1, dtype=np.uint64)
    node_tags = np.arange(1, len(coordinates) + 1, dtype=np.uint64)
    element_type = {dolfinx.mesh.CellType.tetrahedron: 11, dolfinx.mesh.CellType.hexahedron: 12}[
        mesh.topology.cell_type
    ]
    points = basix.create_lattice(mesh.basix_cell(), 4, basix.LatticeType.equispaced, True)
    expression = dolfinx.fem.Expression(ufl.det(ufl.Identity(3) + ufl.grad(displacement)), points)
    sample = deformation_jacobian_range(displacement, cells)
    gmsh_points = points if mesh.topology.cell_type == dolfinx.mesh.CellType.tetrahedron else 2 * points - 1
    validation_cells = np.unique(np.linspace(0, num_cells - 1, min(20, num_cells), dtype=np.int32))
    lengths = np.ptp(coordinates, axis=0)
    origin = coordinates.min(axis=0)
    gmsh.model.add(path.stem)
    entity = gmsh.model.addDiscreteEntity(3)

    def build(x):
        gmsh.model.mesh.addNodes(3, entity, node_tags, x.ravel())
        gmsh.model.mesh.addElementsByType(entity, element_type, element_tags, connectivity.ravel())

    def bounds(kind):
        return gmsh.model.mesh.getElementQualities(element_tags, kind)

    def selected_jacobians():
        return np.array([gmsh.model.mesh.getJacobian(int(i + 1), gmsh_points.ravel())[1] for i in validation_cells])

    try:
        build(coordinates)
        center = [0.25, 0.25, 0.25] if element_type == 11 else [0, 0, 0]
        determinants = gmsh.model.mesh.getJacobians(element_type, center, entity)[1]
        reversed_tags = element_tags[np.asarray(determinants) < 0]
        if len(reversed_tags):
            gmsh.model.mesh.reverseElements(reversed_tags)
            _, ordered_nodes = gmsh.model.mesh.getElementsByType(element_type, entity)
            connectivity = ordered_nodes.reshape(connectivity.shape).copy()
        reference_min = bounds("minDetJac")
        reference_max = bounds("maxDetJac")
        if not np.all(reference_min > 0):
            raise ValueError("Gmsh could not establish positive rest-element Jacobians")
        reference_samples = selected_jacobians()
        results = []
        for amplitude in amplitudes:

            def synthetic(x, amplitude=amplitude):
                scaled = (x - origin[:, None]) / lengths[:, None]
                shape = np.prod(np.sin(np.pi * scaled), axis=0)
                return amplitude * lengths[:, None] * shape

            displacement.interpolate(synthetic)
            sample_min, sample_max = sample()

            def rebuild():
                deformed = coordinates + displacement.x.array.reshape(-1, 3)
                gmsh.model.mesh.clear()
                build(deformed)

            rebuild()
            ratio_samples = selected_jacobians() / reference_samples
            dolfinx_samples = expression.eval(mesh, validation_cells).reshape(len(validation_cells), -1)
            # Reversing a cell permutes lattice points, so compare sorted values.
            np.testing.assert_allclose(
                np.sort(ratio_samples, axis=1), np.sort(dolfinx_samples, axis=1), atol=2e-10, rtol=2e-10
            )
            deformed_min = bounds("minDetJac")
            ratio_lower = deformed_min / np.where(deformed_min >= 0, reference_max, reference_min)
            lower = float(ratio_lower.min())
            if lower > sample_min + 1e-9:
                raise ValueError("Gmsh lower bound exceeds sampled min det(F)")

            def quality_both():
                bounds("minDetJac")
                bounds("maxDetJac")

            def bulk_min():
                rebuild()
                bounds("minDetJac")

            def bulk_both():
                rebuild()
                quality_both()

            row = {
                "amplitude": amplitude,
                "sample_min_F": sample_min,
                "sample_max_F": sample_max,
                "gmsh_min_F_lower": lower,
                "sampling": timed(sample, repeats),
                "gmsh_min_only": timed(lambda: bounds("minDetJac"), repeats),
                "gmsh_min_max": timed(quality_both, repeats),
                "bulk_rebuild": timed(rebuild, repeats),
                "bulk_min": timed(bulk_min, repeats),
                "bulk_min_max": timed(bulk_both, repeats),
            }
            results.append(row)
            print(json.dumps({"mesh": path.name, "cells": num_cells, **row}), flush=True)
        # A 3D volume determinant must retain the sign of a uniform reflection.
        reflected = coordinates.copy()
        reflected[:, 0] *= -1
        gmsh.model.mesh.clear()
        build(reflected)
        reflected_min = bounds("minDetJac")
        np.testing.assert_allclose(reflected_min, -reference_max, atol=1e-12, rtol=1e-9)
        return {
            "mesh": str(path),
            "cell_type": mesh.topology.cell_type.name,
            "cells": num_cells,
            "nodes": len(coordinates),
            "sample_points_per_cell": len(points),
            "rest_jacobian_max_ratio": float(np.max(reference_max / reference_min)),
            "validation_cells": len(validation_cells),
            "reflection_sign_check": "passed",
            "results": results,
        }
    finally:
        gmsh.model.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meshes", nargs="+", type=Path)
    parser.add_argument("--amplitudes", nargs="+", type=float, default=[0.03, 0.20])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1:
        parser.error("Run this serial microbenchmark without an MPI launcher")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    gmsh.initialize([], readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    report = {"host": platform.node(), "gmsh": gmsh.__version__, "dolfinx": dolfinx.__version__, "meshes": []}
    try:
        for path in args.meshes:
            report["meshes"].append(benchmark(path, args.amplitudes, args.repeats))
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()

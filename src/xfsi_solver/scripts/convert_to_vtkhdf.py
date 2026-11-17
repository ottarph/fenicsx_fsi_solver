# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Convert the output files and checkpoints of earlier runs, in ADIOS2 ``.bp`` files, to VTKHDF.

    python -m xfsi_solver.scripts.convert_to_vtkhdf <mesh> <source> [<source> ...] [--output-dir DIR]

``mesh`` is the XDMF mesh file the runs were made on, with its cell tags, which the old files do
not hold. Each ``source`` is a ``.bp`` file, or a directory searched for ``.bp`` files (a run
directory, a checkpoint directory, a snapshot directory, ...). Each file is converted by its kind:

- **VTX output** of ``dolfinx.io.VTXWriter`` (``output/pv/<run>.bp``, ``<run>_p.bp``) becomes a
  VTKHDF time series as ``xfsi_solver.tools.vtkhdf.VTKHDFWriter`` writes it: every time of the file,
  the functions at the nodes of the mesh, functions on a submesh (p on the fluid cells) zero outside
  it, and the cell tags and owning ranks as cell data. The files of one run that the old solvers wrote
  apart, ``<name>.bp`` with ``<name>_p.bp`` and ``<name>_solid.bp`` (also when restarted,
  ``<name>_from_t<t>.bp`` with ``<name>_p_from_t<t>.bp``), are merged into one, ``<name>.vtkhdf``, if
  they hold the same times.
- **Checkpoints** of ``xfsi_solver.tools.checkpoint`` (``checkpoint_<i>.bp``, also in snapshot
  directories) become VTKHDF checkpoints, ``checkpoint_<i>.vtkhdf``, with the same attributes (time,
  step, ...), from which a run restarts or starts as from one it wrote itself. Checkpoints without
  them, which a run killed while writing left incomplete, are skipped.

Other ``.bp`` files are skipped with a message. The converted files are written next to the old
ones, or, with ``--output-dir``, under ``DIR`` with the same paths relative to each source directory
(a ``source`` file goes to ``DIR`` itself). The old files are not changed. A converted file that
exists is left as it is, unless ``--overwrite`` is given.

The VTX files hold, for each MPI rank of the run that wrote them, the coordinates of the nodes of the
function space and the values there. They are matched to the dofs of a continuous Lagrange space of
degree 1 or 2 on the mesh, or on the submesh of the cells with one of its cell tags, by these
coordinates, so the conversion may run on any number of MPI ranks, and finds which space, and so
which submesh, each file is on. A checkpoint holds the dofs of each function on the mesh in its input
cell order, and its space is found from the number of dofs. Only functions of up to 2 degrees on the
same mesh as ``mesh`` are converted, which is what the solvers write; a file of another mesh raises
``ValueError``.
"""

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import adios2
import dolfinx
import io4dolfinx
import numpy as np
from mpi4py import MPI
from scipy.spatial import cKDTree

from xfsi_solver.tools import vtkhdf
from xfsi_solver.tools.checkpoint import _ATTRIBUTE_KEYS, _ATTRIBUTES

# The variables of a VTX file that are not functions
_VTX_MESH_VARIABLES = {
    "NumberOfEntities",
    "NumberOfNodes",
    "connectivity",
    "geometry",
    "step",
    "types",
    "vtkGhostType",
    "vtkOriginalPointIds",
}
# Files the old solvers wrote apart from <name>.bp, with functions on other (sub)meshes
_PARTNER_SUFFIXES = ("_p", "_solid")
_DEGREES = (1, 2)
# Nodes closer than this, relative to the size of the domain, are the same node
_COORDINATE_RTOL = 1e-8


def _on_rank_0(comm: MPI.Comm, read):
    """``read()`` on rank 0, sent to every rank; an exception on rank 0 is raised on every rank."""
    result, error = None, None
    if comm.rank == 0:
        try:
            result = read()
        except Exception as e:  # noqa: BLE001
            error = e
    result, error = comm.bcast((result, error), root=0)
    if error is not None:
        raise error
    return result


def file_kind(path: str | Path, comm: MPI.Comm) -> str | None:
    """``"vtx"`` for a VTX file, ``"checkpoint"`` for an io4dolfinx checkpoint, ``None`` for other files."""

    def read():
        with adios2.FileReader(str(path)) as f:
            if "vtk.xml" in f.available_attributes():
                return "vtx"
            variables = f.available_variables()
            if "Topology" in variables and any(name.endswith("_values") for name in variables):
                return "checkpoint"
        return None

    try:
        return _on_rank_0(comm, read)
    except Exception:  # noqa: BLE001
        return None


class MeshData:
    """The mesh of the XDMF file ``mesh_path``, its cell tags, and the submeshes of the cells of each tag."""

    def __init__(self, mesh_path: str | Path, comm: MPI.Comm = MPI.COMM_WORLD, cell_tags_name: str = "Cell tags"):
        with dolfinx.io.XDMFFile(comm, str(mesh_path), "r") as infile:
            self.mesh = infile.read_mesh()
            self.cell_tags = infile.read_meshtags(self.mesh, name=cell_tags_name)
        self.comm = comm
        self.tags = sorted({int(v) for vals in comm.allgather(np.unique(self.cell_tags.values)) for v in vals})
        self._submeshes = {}
        self._layouts = {}

    @property
    def num_cells(self) -> int:
        return self.mesh.topology.index_map(self.mesh.topology.dim).size_global

    def submesh(self, tag: int | None) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.EntityMap | None]:
        """The mesh (``tag=None``), or the submesh of the cells tagged ``tag``, with its entity map."""
        if tag is None:
            return self.mesh, None
        if tag not in self._submeshes:
            tdim = self.mesh.topology.dim
            submesh, entity_map, _, _ = dolfinx.mesh.create_submesh(self.mesh, tdim, self.cell_tags.find(tag))
            self._submeshes[tag] = (submesh, entity_map)
        return self._submeshes[tag]

    def layout(self, tags: Sequence[int | None]) -> vtkhdf.PointDataLayout:
        """The layout of VTKHDF files with functions on the mesh and on the submeshes of ``tags``."""
        key = tuple(sorted(tag for tag in set(tags) if tag is not None))
        if key not in self._layouts:
            self._layouts[key] = vtkhdf.PointDataLayout(self.mesh, [self.submesh(tag) for tag in key])
        return self._layouts[key]


class VTXFile:
    """A VTX file of ``dolfinx.io.VTXWriter``: its times, functions, and the nodes and values of each rank."""

    def __init__(self, path: str | Path, comm: MPI.Comm):
        self.path = Path(path)

        def read():
            with adios2.FileReader(str(self.path)) as f:
                num_steps = f.num_steps()
                names = sorted(set(f.available_variables()) - _VTX_MESH_VARIABLES)
                times = np.array([f.read("step", step_selection=[s, 1]).item() for s in range(num_steps)])
                nodes_per_block = f.read("NumberOfNodes", step_selection=[0, 1])
                # The mesh is in every step, or, with mesh_policy=reuse, in the first only
                geometry_steps = int(f.available_variables()["geometry"]["AvailableStepsCount"])
            return names, times, nodes_per_block, geometry_steps

        self.names, self.times, nodes_per_block, geometry_steps = _on_rank_0(comm, read)
        # A block of each rank of the run; those of ranks with no cell of the (sub)mesh are empty, and
        # adios2 cannot read them
        self._blocks = [b for b, num_nodes in enumerate(nodes_per_block) if num_nodes > 0]
        self.nodes = self._read_blocks("geometry", 0)
        if geometry_steps > 1 and not np.array_equal(self._read_blocks("geometry", geometry_steps - 1), self.nodes):
            raise ValueError(f"{self.path} holds a mesh that moves, which is not converted")

    def _read_blocks(self, name: str, step: int) -> np.ndarray:
        """The blocks of all ranks of the variable ``name`` at step ``step``, one after the other, as rows."""
        with adios2.FileReader(str(self.path)) as f:
            blocks = [f.read(name, block_id=b, step_selection=[step, 1]) for b in self._blocks]
        return np.concatenate([block.reshape(len(block), -1) for block in blocks])

    def values(self, name: str, step: int) -> np.ndarray:
        """The values of the function ``name`` at the nodes at step ``step``, a row for each node."""
        return self._read_blocks(name, step)


def _output_times(path: Path, comm: MPI.Comm) -> tuple[float, ...] | None:
    """The times in the VTX file ``path``, or None if they cannot be read, which its conversion then reports."""

    def read():
        with adios2.FileReader(str(path)) as f:
            return tuple(f.read("step", step_selection=[s, 1]).item() for s in range(f.num_steps()))

    try:
        return _on_rank_0(comm, read)
    except Exception:  # noqa: BLE001
        return None


def _num_unique(points: np.ndarray, tol: float) -> int:
    """The number of distinct points in ``points``, where points closer than ``tol`` are the same."""
    pairs = cKDTree(points).query_pairs(tol, output_type="ndarray")
    return len(points) - len(np.unique(pairs[:, 1])) if len(pairs) else len(points)


def _match_space(
    data: MeshData, vtx: VTXFile, num_components: int
) -> tuple[int | None, dolfinx.fem.FunctionSpace, np.ndarray]:
    """The tag of the (sub)mesh, the Lagrange space on it whose dofs are the nodes of ``vtx``, and the node of
    each local dof of it, owned and ghost.

    The nodes of each rank of the run that wrote the file include its ghosts, so a node may be in the
    file more than once; any copy has the same value.
    """
    comm = data.comm
    tol = _COORDINATE_RTOL * max(float(np.ptp(vtx.nodes, axis=0).max()), 1.0e-300)
    num_nodes = _num_unique(vtx.nodes, tol)
    shape = () if num_components == 1 else (num_components,)
    tree = None
    for tag in [None, *data.tags]:
        mesh, _ = data.submesh(tag)
        for degree in _DEGREES:
            V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, shape))
            if V.dofmap.index_map.size_global != num_nodes:
                continue
            if tree is None:
                tree = cKDTree(vtx.nodes)
            distance, node = tree.query(V.tabulate_dof_coordinates())
            if comm.allreduce(bool(np.all(distance <= tol)), op=MPI.LAND):
                return tag, V, node
    raise ValueError(
        f"The {num_nodes} nodes of {vtx.path} are not the dofs of a Lagrange space of degree {_DEGREES} on the mesh "
        f"or on the cells of one of its cell tags {data.tags}; was it written on another mesh?"
    )


def convert_output(data: MeshData, paths: Sequence[str | Path], out_path: str | Path) -> None:
    """Write the functions of the VTX files ``paths``, which hold the same times, to the VTKHDF file ``out_path``."""
    comm = data.comm
    gdim = data.mesh.geometry.dim
    files = [VTXFile(path, comm) for path in paths]
    times = files[0].times
    functions = []  # (VTX file, function, local dof -> node)
    tags = []
    for vtx in files:
        if not np.allclose(vtx.times, times, rtol=1e-12, atol=0.0):
            raise ValueError(f"{vtx.path} holds other times than {files[0].path}")
        if not vtx.names:
            continue
        num_components = vtx.values(vtx.names[0], 0).shape[1]
        if num_components == 3 and gdim == 2:
            num_components = 2  # VTXWriter pads vectors on 2D meshes to 3 components for VTK
        if num_components not in (1, gdim):
            raise NotImplementedError(f"{vtx.path} holds values of {num_components} components, not scalars or vectors")
        tag, V, node = _match_space(data, vtx, num_components)
        tags.append(tag)
        for name in vtx.names:
            functions.append((vtx, dolfinx.fem.Function(V, name=name), node))
    names = [f.name for _, f, _ in functions]
    if len(set(names)) < len(names):
        raise ValueError(f"The files {[str(p) for p in paths]} hold functions of the same name: {names}")

    layout = data.layout(tags)
    tmp_path = _partial(out_path)
    for step, t in enumerate(times):
        for vtx, f, node in functions:
            bs = f.function_space.dofmap.index_map_bs
            f.x.array.reshape(-1, bs)[:] = vtx.values(f.name, step)[node, :bs]
        if step == 0:
            layout.write_mesh(tmp_path, t, data.cell_tags)
        layout.write_functions(tmp_path, [f for _, f, _ in functions], t)
    _finish(tmp_path, out_path, comm)


def _checkpoint_spaces(path: Path, data: MeshData) -> dict[str, tuple[int, int]]:
    """The global number of values and of dofs per cell of each function in the checkpoint ``path``."""

    def read():
        with adios2.FileReader(str(path)) as f:
            variables = f.available_variables()
            sizes = {}
            for name, info in variables.items():
                function = name.removesuffix("_values")
                if name.endswith("_values") and f"{function}_dofmap" in variables:
                    num_values = int(info["Shape"].split(",")[0])
                    dofmap_size = int(variables[f"{function}_dofmap"]["Shape"].split(",")[0])
                    sizes[function] = (num_values, dofmap_size)
            return sizes

    sizes = _on_rank_0(data.comm, read)
    return {name: (num_values, dofmap_size // data.num_cells) for name, (num_values, dofmap_size) in sizes.items()}


def convert_checkpoint(data: MeshData, path: str | Path, out_path: str | Path) -> bool:
    """Write the checkpoint ``path`` as the VTKHDF checkpoint ``out_path``; False, and nothing written, if
    ``path`` is incomplete.

    Functions on a submesh are stored on the whole mesh, with zero outside the submesh, so they are
    written as functions on the mesh, which is how VTKHDF files store them too.
    """
    comm = data.comm
    path = Path(path)
    attrs = io4dolfinx.read_attributes(path, comm, _ATTRIBUTES)
    if not _ATTRIBUTE_KEYS <= attrs.keys():
        return False
    if int(attrs["num_cells"][0]) != data.num_cells:
        raise ValueError(f"{path} was written on a mesh with {int(attrs['num_cells'][0])} cells, not {data.num_cells}")
    t = float(attrs["t"][0])
    gdim = data.mesh.geometry.dim
    functions = []
    for name, (num_values, dofs_per_cell) in sorted(_checkpoint_spaces(path, data).items()):
        candidates = [
            V
            for degree in _DEGREES
            for shape in [(), (gdim,)]
            if (V := dolfinx.fem.functionspace(data.mesh, ("Lagrange", degree, shape)))
            and V.dofmap.index_map.size_global * V.dofmap.index_map_bs == num_values
            and V.dofmap.dof_layout.num_dofs * V.dofmap.index_map_bs == dofs_per_cell
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"{path} holds {name!r} with {num_values} values and {dofs_per_cell} per cell, which is "
                f"{'not a' if not candidates else 'more than one'} Lagrange space of degree {_DEGREES} on the mesh"
            )
        times = np.asarray(io4dolfinx.read_timestamps(path, comm, name))
        if not np.any(np.isclose(times, t, rtol=1e-12, atol=0.0)):
            raise ValueError(f"{path} holds {name!r} at t = {times}, not at the time of the checkpoint, {t}")
        f = dolfinx.fem.Function(candidates[0], name=name)
        io4dolfinx.read_function(path, f, time=t, name=name)
        functions.append(f)

    layout = data.layout([])
    tmp_path = _partial(out_path)
    layout.write_mesh(tmp_path, t, data.cell_tags)
    layout.write_functions(tmp_path, functions, t)
    # Last, as in a checkpoint a run writes: a file without them is incomplete
    vtkhdf.write_attributes(tmp_path, comm, _ATTRIBUTES, attrs)
    _finish(tmp_path, out_path, comm)
    return True


def _partial(out_path: str | Path) -> Path:
    """The file written before it is complete, then renamed to ``out_path``."""
    out_path = Path(out_path)
    return out_path.with_name(f"{out_path.stem}.partial{out_path.suffix}")


def _finish(tmp_path: Path, out_path: str | Path, comm: MPI.Comm) -> None:
    comm.Barrier()
    if comm.rank == 0:
        tmp_path.replace(out_path)
    comm.Barrier()


def find_bp_files(source: str | Path) -> list[Path]:
    """``source``, if it is a ``.bp`` file, else the ``.bp`` files under the directory ``source``."""
    source = Path(source)
    if source.suffix == ".bp":
        return [source]
    found = []
    for path in sorted(source.rglob("*.bp")):
        # Not the directories inside an ADIOS2 file
        if not any(parent.suffix == ".bp" for parent in path.relative_to(source).parents):
            found.append(path)
    return found


def group_outputs(paths: Sequence[Path]) -> list[list[Path]]:
    """The VTX files ``paths`` in groups to merge: ``<name>.bp`` first, then its partners such as ``<name>_p.bp``."""
    paths = list(paths)
    known = set(paths)
    groups = {path: [path] for path in paths}
    for path in paths:
        match = re.fullmatch(r"(.*?)(_from_t\d+\.\d+)?", path.stem)
        base, restart = match.group(1), match.group(2) or ""
        for suffix in _PARTNER_SUFFIXES:
            main = path.with_name(f"{base.removesuffix(suffix)}{restart}.bp")
            if base.endswith(suffix) and main in known and main != path:
                groups[main].append(groups.pop(path)[0])
                break
    return list(groups.values())


def convert(
    mesh_path: str | Path,
    sources: Sequence[str | Path],
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    comm: MPI.Comm = MPI.COMM_WORLD,
    cell_tags_name: str = "Cell tags",
) -> list[Path]:
    """Convert the ``.bp`` files in ``sources`` to VTKHDF, as the module describes; returns the files written.

    A file that fails to convert is reported, and the others are still converted; then ``RuntimeError``
    is raised at the end, with the reasons.
    """
    data = MeshData(mesh_path, comm, cell_tags_name)

    def out_path(path: Path, source: Path) -> Path:
        if output_dir is None:
            return path.with_suffix(vtkhdf.SUFFIX)
        relative = Path(path.name) if source.suffix == ".bp" else path.relative_to(source)
        return (Path(output_dir) / relative).with_suffix(vtkhdf.SUFFIX)

    jobs = []  # (kind, the files, converted file)
    outputs = {}  # VTX file -> the source it was found in
    for source in map(Path, sources):
        if not source.exists():
            raise FileNotFoundError(source)
        for path in find_bp_files(source):
            kind = file_kind(path, comm)
            if kind == "vtx":
                outputs.setdefault(path, source)
            elif kind == "checkpoint":
                jobs.append(("checkpoint", [path], out_path(path, source)))
            elif comm.rank == 0:
                print(f"Skipping {path}: neither VTX output nor a checkpoint", flush=True)
    # Grouped over all sources, so that <name>.bp and <name>_p.bp given apart are merged too
    for group in group_outputs(list(outputs)):
        if len(group) > 1 and len({_output_times(path, comm) for path in group}) > 1:
            if comm.rank == 0:
                print(f"Not merging {[str(p) for p in group]}: they hold different times", flush=True)
            jobs += [("vtx", [path], out_path(path, outputs[path])) for path in group]
        else:
            jobs.append(("vtx", group, out_path(group[0], outputs[group[0]])))

    written, failed = [], []
    for kind, paths, target in jobs:
        names = " + ".join(str(path) for path in paths)
        if target.exists() and not overwrite:
            if comm.rank == 0:
                print(f"Skipping {names}: {target} exists", flush=True)
            continue
        if comm.rank == 0:
            target.parent.mkdir(parents=True, exist_ok=True)
        comm.Barrier()
        try:
            if kind == "vtx":
                convert_output(data, paths, target)
            elif not convert_checkpoint(data, paths[0], target):
                if comm.rank == 0:
                    print(f"Skipping {names}: an incomplete checkpoint", flush=True)
                continue
        except Exception as e:  # noqa: BLE001
            failed.append(f"{names}: {e}")
            if comm.rank == 0:
                print(f"Could not convert {names}: {e}", file=sys.stderr, flush=True)
            continue
        written.append(target)
        if comm.rank == 0:
            print(f"Wrote {target} from {names}", flush=True)
    if failed:
        raise RuntimeError("Could not convert\n" + "\n".join(failed))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mesh", help="the XDMF mesh file, with cell tags, that the runs were made on")
    parser.add_argument("sources", nargs="+", help=".bp files, or directories to search for them")
    parser.add_argument("--output-dir", help="write under this directory instead of next to the old files")
    parser.add_argument("--overwrite", action="store_true", help="replace converted files that exist")
    parser.add_argument("--cell-tags-name", default="Cell tags", help="the name of the cell tags in the mesh file")
    args = parser.parse_args()
    convert(args.mesh, args.sources, args.output_dir, args.overwrite, cell_tags_name=args.cell_tags_name)


if __name__ == "__main__":
    main()

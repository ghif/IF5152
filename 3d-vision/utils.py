"""
Shared utilities for IF5152 3D Vision Laboratory.
Supports NVIDIA CUDA, Apple Silicon (MPS), and CPU.
Provides synthetic 3D shape generators (voxels, point clouds, SDFs, meshes)
and headless matplotlib visualization helpers.
"""

import os
from typing import Optional, Tuple
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
DEFAULT_SAMPLES_DIR = os.path.join(SCRIPT_DIR, "samples")
DEFAULT_CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")

MODELNET10_CATEGORIES = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]


def download_and_extract_modelnet10(data_dir: Optional[str] = None) -> str:
    """
    Downloads ModelNet10.zip (Princeton 3D ShapeNets) and extracts into data_dir.
    Returns the path to the ModelNet10 directory.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    modelnet_dir = os.path.join(data_dir, "ModelNet10")
    if os.path.exists(modelnet_dir) and any(os.scandir(modelnet_dir)):
        return modelnet_dir

    url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    zip_path = os.path.join(data_dir, "ModelNet10.zip")
    print(f"[ModelNet10] Downloading from {url} to {zip_path}...")

    import urllib.request
    from tqdm import tqdm

    class DownloadProgressBar(tqdm):
        def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="ModelNet10.zip") as t:
        urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)

    print(f"[ModelNet10] Extracting {zip_path} to {data_dir}...")
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    if os.path.exists(zip_path):
        os.remove(zip_path)
    print(f"[ModelNet10] Ready at {modelnet_dir}")
    return modelnet_dir


def load_off_mesh(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads an Object File Format (.off) 3D mesh.
    Centers and scales the vertices to a unit bounding sphere.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f.read().splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        raise ValueError(f"Empty OFF file: {filepath}")

    first = lines[0]
    if first == "OFF":
        counts = lines[1].split()
        start = 2
    elif first.startswith("OFF"):
        counts = first[3:].strip().split()
        start = 1
    else:
        counts = first.split()
        start = 1

    n_v, n_f = int(counts[0]), int(counts[1])
    verts = [[float(x) for x in lines[i].split()[:3]] for i in range(start, start + n_v)]
    verts = np.array(verts, dtype=np.float32)

    faces = []
    for i in range(start + n_v, start + n_v + n_f):
        parts = [int(x) for x in lines[i].split()]
        k = parts[0]
        idx = parts[1 : 1 + k]
        for t in range(1, k - 1):
            faces.append([idx[0], idx[t], idx[t + 1]])
    faces = np.array(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64)

    if len(verts) > 0:
        center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
        verts = verts - center
        scale = np.max(np.linalg.norm(verts, axis=-1)) + 1e-8
        verts = (verts / scale) * 0.8

    return verts, faces


def render_mesh_multiview(
    verts: np.ndarray,
    faces: np.ndarray,
    num_views: int = 12,
    img_size: int = 64,
    num_points: int = 3500,
) -> np.ndarray:
    """
    Renders V multi-view depth/silhouette images around a 3D mesh by sampling
    surface points and projecting onto virtual camera planes on an elevation 30-deg orbit.
    Returns:
        views: (num_views, 1, img_size, img_size) float32 in [0, 1]
    """
    if len(verts) == 0:
        return np.zeros((num_views, 1, img_size, img_size), dtype=np.float32)

    if len(faces) > 0:
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=-1)
        tot_area = areas.sum()
        if tot_area > 1e-8:
            probs = areas / tot_area
            idx = np.random.choice(len(faces), size=num_points, p=probs)
            r1 = np.sqrt(np.random.rand(num_points, 1).astype(np.float32))
            r2 = np.random.rand(num_points, 1).astype(np.float32)
            pts = (1 - r1) * v0[idx] + r1 * (1 - r2) * v1[idx] + r1 * r2 * v2[idx]
        else:
            idx = np.random.choice(len(verts), size=num_points, replace=True)
            pts = verts[idx]
    else:
        idx = np.random.choice(len(verts), size=num_points, replace=True)
        pts = verts[idx]

    elev = np.radians(30.0)
    views = []
    for v in range(num_views):
        az = 2.0 * np.pi * v / num_views
        c, s = np.cos(az), np.sin(az)
        ce, se = np.cos(elev), np.sin(elev)
        R = np.array([
            [-s, 0, c],
            [-c * se, ce, -s * se],
            [-c * ce, -se, -s * ce],
        ], dtype=np.float32)
        pc = pts @ R.T

        u = ((pc[:, 0] + 0.9) / 1.8 * (img_size - 1)).astype(np.int32)
        v_coord = ((-pc[:, 1] + 0.9) / 1.8 * (img_size - 1)).astype(np.int32)
        valid = (u >= 0) & (u < img_size) & (v_coord >= 0) & (v_coord < img_size)
        u, v_coord, z = u[valid], v_coord[valid], pc[valid, 2]

        img = np.zeros((img_size, img_size), dtype=np.float32)
        if len(z) > 0:
            z_min, z_max = z.min(), z.max()
            depth_val = 0.35 + 0.65 * (z - z_min) / (z_max - z_min + 1e-6)
            for du, dv in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                uu = np.clip(u + du, 0, img_size - 1)
                vv = np.clip(v_coord + dv, 0, img_size - 1)
                np.maximum.at(img, (vv, uu), depth_val * 0.95)
        views.append(img[None, ...])

    return np.array(views, dtype=np.float32)


def sample_mesh_surface_points(
    verts: np.ndarray,
    faces: np.ndarray,
    num_points: int = 1024,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Uniformly samples N points from the surface of a triangle mesh proportional to face areas
    using random barycentric coordinates.
    Returns:
        (num_points, 3) float32 numpy array.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(verts) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    if len(faces) == 0:
        idx = rng.choice(len(verts), size=num_points, replace=True)
        return verts[idx].astype(np.float32)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=-1)
    tot_area = areas.sum()

    if tot_area <= 1e-8:
        idx = rng.choice(len(verts), size=num_points, replace=True)
        return verts[idx].astype(np.float32)

    probs = areas / tot_area
    face_idx = rng.choice(len(faces), size=num_points, p=probs)

    r1 = rng.random(num_points, dtype=np.float32)
    r2 = rng.random(num_points, dtype=np.float32)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2

    pts = (
        u[:, None] * v0[face_idx]
        + v[:, None] * v1[face_idx]
        + w[:, None] * v2[face_idx]
    )
    return pts.astype(np.float32)


def voxelize_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    resolution: int = 24,
    num_dense_points: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a continuous 3D triangle mesh into a binary (resolution x resolution x resolution)
    voxel occupancy grid and an orthographic depth/silhouette 2D projection.
    Returns:
        voxel_grid: (resolution, resolution, resolution) float32 in {0.0, 1.0}
        projection: (1, resolution, resolution) float32
    """
    pts = sample_mesh_surface_points(verts, faces, num_points=num_dense_points, rng=rng)
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    coords = np.clip(
        np.floor((pts + 0.9) / 1.8 * resolution).astype(np.int32),
        0,
        resolution - 1,
    )
    grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0

    proj = grid.mean(axis=2, keepdims=True)  # (V, V, 1)
    proj_tensor = proj.transpose(2, 0, 1)    # (1, V, V)
    return grid, proj_tensor


def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Selects the optimal compute device.
    Priority if None: CUDA -> MPS (Apple Silicon) -> CPU.
    """
    if preferred:
        pref = preferred.lower()
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif pref == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif pref == "cpu":
            return torch.device("cpu")
        print(f"[Warning] Requested device '{preferred}' unavailable. Falling back.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dirs() -> None:
    """Create default samples and checkpoints directories."""
    os.makedirs(DEFAULT_SAMPLES_DIR, exist_ok=True)
    os.makedirs(DEFAULT_CHECKPOINT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Analytical Signed Distance Functions (SDFs) for 3D Primitives
# ---------------------------------------------------------------------------

def sdf_sphere(points: np.ndarray, radius: float = 0.6) -> np.ndarray:
    """SDF of a sphere centered at origin: ||p|| - r."""
    return np.linalg.norm(points, axis=-1) - radius


def sdf_box(points: np.ndarray, half_extents: Tuple[float, float, float] = (0.5, 0.5, 0.5)) -> np.ndarray:
    """Exact SDF of an axis-aligned box centered at origin."""
    b = np.array(half_extents, dtype=np.float32)
    q = np.abs(points) - b
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def sdf_torus(points: np.ndarray, major_r: float = 0.55, minor_r: float = 0.22) -> np.ndarray:
    """Exact SDF of a torus lying in the XZ plane."""
    q_x = np.linalg.norm(points[..., [0, 2]], axis=-1) - major_r
    q_y = points[..., 1]
    return np.sqrt(q_x ** 2 + q_y ** 2) - minor_r


def sdf_cylinder(points: np.ndarray, radius: float = 0.4, half_height: float = 0.6) -> np.ndarray:
    """Exact SDF of a vertical cylinder along Y axis."""
    d_xz = np.linalg.norm(points[..., [0, 2]], axis=-1) - radius
    d_y = np.abs(points[..., 1]) - half_height
    outside = np.sqrt(np.maximum(d_xz, 0.0) ** 2 + np.maximum(d_y, 0.0) ** 2)
    inside = np.minimum(np.maximum(d_xz, d_y), 0.0)
    return outside + inside


def sdf_csg_chair(points: np.ndarray) -> np.ndarray:
    """
    Constructive Solid Geometry (CSG) example: a simple chair built from
    Union of box primitives (seat + backrest + 4 legs).
    """
    seat = sdf_box(points - np.array([0.0, -0.05, 0.0]), (0.42, 0.07, 0.42))
    back = sdf_box(points - np.array([0.0, 0.35, -0.35]), (0.42, 0.40, 0.07))
    leg_offsets = [
        [-0.32, -0.40, -0.32],
        [0.32, -0.40, -0.32],
        [-0.32, -0.40, 0.32],
        [0.32, -0.40, 0.32],
    ]
    d = np.minimum(seat, back)
    for off in leg_offsets:
        leg = sdf_box(points - np.array(off), (0.07, 0.35, 0.07))
        d = np.minimum(d, leg)
    return d


SHAPE_NAMES = ["sphere", "cube", "torus", "cylinder"]


def evaluate_shape_sdf(shape_name: str, points: np.ndarray) -> np.ndarray:
    """Evaluate SDF for a named 3D primitive."""
    name = shape_name.lower()
    if name == "sphere":
        return sdf_sphere(points)
    elif name in ("cube", "box"):
        return sdf_box(points)
    elif name == "torus":
        return sdf_torus(points)
    elif name == "cylinder":
        return sdf_cylinder(points)
    elif name == "chair":
        return sdf_csg_chair(points)
    raise ValueError(f"Unknown shape_name: {shape_name}")


def sample_surface_points(
    shape_name: str,
    num_points: int = 1024,
    rng: Optional[np.random.Generator] = None,
    noise_std: float = 0.0,
) -> np.ndarray:
    """
    Rejection-sample points near the zero-level set of an analytical SDF,
    then project them onto the surface via 2 Newton-like SDF steps.
    """
    if rng is None:
        rng = np.random.default_rng()

    collected = []
    total = 0
    while total < num_points:
        candidates = rng.uniform(-1.0, 1.0, size=(num_points * 4, 3)).astype(np.float32)
        sdf_vals = evaluate_shape_sdf(shape_name, candidates)
        mask = np.abs(sdf_vals) < 0.08
        pts = candidates[mask]
        if len(pts) > 0:
            # Finite-difference normal projection onto surface
            eps = 1e-3
            for _ in range(3):
                val = evaluate_shape_sdf(shape_name, pts)
                grad = np.zeros_like(pts)
                for axis in range(3):
                    delta = np.zeros_like(pts)
                    delta[:, axis] = eps
                    grad[:, axis] = (
                        evaluate_shape_sdf(shape_name, pts + delta)
                        - evaluate_shape_sdf(shape_name, pts - delta)
                    ) / (2.0 * eps)
                norm = np.linalg.norm(grad, axis=-1, keepdims=True) + 1e-8
                pts = pts - val[:, None] * (grad / norm)
            collected.append(pts)
            total += len(pts)

    result = np.concatenate(collected, axis=0)[:num_points]
    if noise_std > 0.0:
        result = result + rng.normal(0.0, noise_std, size=result.shape).astype(np.float32)
    return result.astype(np.float32)


def generate_voxel_grid(shape_name: str, resolution: int = 24) -> np.ndarray:
    """Generate a binary V x V x V occupancy grid from an analytical SDF."""
    coords = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    grid_pts = np.stack([gx, gy, gz], axis=-1)
    sdf = evaluate_shape_sdf(shape_name, grid_pts)
    return (sdf <= 0.0).astype(np.float32)


def create_icosphere(subdivisions: int = 2, radius: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a closed triangle mesh icosphere (vertices V x 3, faces F x 3).
    Useful as a template mesh for Pixel2Mesh-style deformation.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float32)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)

    for _ in range(subdivisions):
        edge_midpoints = {}
        new_verts = list(verts)
        new_faces = []

        def get_midpoint(i0: int, i1: int) -> int:
            key = (min(i0, i1), max(i0, i1))
            if key in edge_midpoints:
                return edge_midpoints[key]
            mid = 0.5 * (new_verts[i0] + new_verts[i1])
            mid /= np.linalg.norm(mid) + 1e-8
            idx = len(new_verts)
            new_verts.append(mid)
            edge_midpoints[key] = idx
            return idx

        for f in faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.extend([
                [a, ab, ca],
                [b, bc, ab],
                [c, ca, bc],
                [ab, bc, ca],
            ])
        verts = np.array(new_verts, dtype=np.float32)
        faces = np.array(new_faces, dtype=np.int64)

    return verts * radius, faces


# ---------------------------------------------------------------------------
# Matplotlib 3D Visualization Helpers
# ---------------------------------------------------------------------------

def plot_voxels_on_ax(ax, occ: np.ndarray, title: str = "", color: str = "steelblue") -> None:
    """Render a 3D binary occupancy grid on a 3D matplotlib axis."""
    filled = occ > 0.5
    ax.voxels(filled, facecolors=color, edgecolor="k", linewidth=0.2, alpha=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def plot_pointcloud_on_ax(ax, pts: np.ndarray, title: str = "", color: str = "royalblue", s: float = 4.0) -> None:
    """Render a 3D point cloud on a 3D matplotlib axis."""
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=pts[:, 1], cmap="viridis", s=s, alpha=0.85)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def plot_mesh_on_ax(ax, verts: np.ndarray, faces: np.ndarray, title: str = "", color: str = "lightcoral") -> None:
    """Render a 3D triangle mesh on a 3D matplotlib axis."""
    tris = verts[faces]
    # Reorder (x, z, y) so Y is up in matplotlib 3D
    tris_vis = tris[..., [0, 2, 1]]
    poly = Poly3DCollection(tris_vis, alpha=0.85, facecolor=color, edgecolor="k", linewidth=0.25)
    ax.add_collection3d(poly)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

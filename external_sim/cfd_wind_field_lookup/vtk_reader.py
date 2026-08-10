from pathlib import Path
import numpy as np
import pyvista as pv


class VtkReader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.tail_path = Path(r"fluid_blocks") / Path(r"fluid_1.vtk")
        self._is_outside_mesh_warned = False

    def get_validated_mesh(self, mesh):
        """Ensure velocity is available as point data."""
        if "velocity" in mesh.point_data:
            return mesh

        if "velocity" in mesh.cell_data:
            return mesh.cell_data_to_point_data()

        available = (
            list(mesh.point_data.keys())
            + list(mesh.cell_data.keys())
        )
        raise KeyError(
            f"'velocity' not found in mesh. Available: {available}"
        )
         
    def load_mesh_by_wind_velocity(self, wind_velocity: np.ndarray, wall_distance: float | None = None):
        wind_velocity_selection = get_wind_velocity_folder_name(wind_velocity, wall_distance)
        vtk_path = get_file_path(self.base_path / Path(wind_velocity_selection),  self.tail_path)
        mesh = pv.read(vtk_path)
        self.free_stream_velocity = np.asarray(wind_velocity, dtype=float).copy()
        self.mesh = self.fill_empty_point_data(self.get_validated_mesh(mesh), wind_velocity)

    def fill_empty_point_data(self, mesh, wind_velocity: np.ndarray):
        """When the cell is not active, the density info is not assigned and defaults to 0. 
        This method uses this characterstic to detect if a cell is empty."""
        density = np.asarray(mesh.point_data["density"])
        velocity = mesh.point_data["velocity"]
        specific_velocity = wind_velocity
        mask = density < 0.1
        velocity[mask] = specific_velocity
        velocity.VTKObject.Modified()
        mesh.point_data["speed"] = np.linalg.norm(velocity, axis=1)

        mesh.Modified()
        return mesh
    
    def get_velocity_at(self, point: np.ndarray):
        """
        Interpolate velocity at a single 3-D point.
        
        Parameters
        ----------
        point : np.ndarray
            Shape (3,).
            
        Returns
        -------
        np.ndarray
            Velocity vector, shape (3,).
        """
        import warnings
        
        pt = np.asarray(point, dtype=float)
        if pt.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {pt.shape}")
        
        # Single-point query
        query = pv.PolyData(pt.reshape(1, 3))
        sampled = query.sample(self.mesh)
        
        velocity = np.asarray(sampled.point_data["velocity"])[0]
        
        # Check if the point was actually inside the mesh domain
        if "vtkValidPointMask" in sampled.point_data:
            valid = bool(sampled.point_data["vtkValidPointMask"][0])

            if not valid:
                if not self._is_outside_mesh_warned:
                    warnings.warn(
                        "Some query points are outside the mesh domain; "
                        "returning free-stream velocity for out-of-range points.",
                        stacklevel=2,
                    )
                    self._is_outside_mesh_warned = True
                return self.free_stream_velocity.copy()

        return velocity

class FreeStreamReader:
    """Drop-in replacement for VtkReader when no wall is present.
    Always returns the free-stream velocity regardless of query point."""
    def __init__(self, free_stream: np.ndarray):
        self._v = np.asarray(free_stream, dtype=float).copy()

    def get_velocity_at(self, point: np.ndarray) -> np.ndarray:
        return self._v.copy()


def get_file_path(base_dir: str | Path, known_filename: str) -> Path:
    base = Path(base_dir)
    
    # Get all immediate subdirectories
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    
    if len(subdirs) != 1:
        raise ValueError(
            f"Expected exactly one subfolder in {base}, found {len(subdirs)}"
        )
    
    file_path = subdirs[0] / known_filename
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    return file_path

def get_wind_velocity_folder_name(wind_velocity: np.ndarray, wall_distance: float | None = None) -> str:
    """Generate a folder name from a 3D wind velocity vector and optional wall distance.

    Parameters
    ----------
    wind_velocity : np.ndarray
        Shape (3,) free-stream velocity [x, y, z].
    wall_distance : float | None
        Wall x-coordinate (signed, metres) — typically negative when the wall
        is in the -x direction (e.g. -0.5).  If None the folder name contains
        only the wind velocity components, preserving backward compatibility
        with files generated before wall distance was encoded.
    """
    wind_velocity = np.asarray(wind_velocity)

    if wind_velocity.shape != (3,):
        raise ValueError(f"wind_velocity must have shape (3,), got {wind_velocity.shape}")

    def make_sign_symbol(value):
        return "n" if value < 0 else "p"

    def encode_float(mag) -> str:
        """Clean float string: 1.0 → '1', 0.5 → '0.5'."""
        if isinstance(mag, (float, np.floating)):
            if mag.is_integer():
                return str(int(mag))
            return f"{mag:.6g}"  # avoids 1.0000000000000002 noise
        return str(mag)

    components = ["x", "y", "z"]
    parts = []

    for value, component in zip(wind_velocity, components):
        sign = make_sign_symbol(value)
        parts.append(f"{component}{sign}{encode_float(abs(value))}")

    if wall_distance is not None:
        sign = make_sign_symbol(wall_distance)
        parts.append(f"d{sign}{encode_float(abs(wall_distance))}")

    return "_".join(parts)
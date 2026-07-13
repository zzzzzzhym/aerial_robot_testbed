from pathlib import Path
import numpy as np
import pyvista as pv


class VtkReader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.tail_path = Path(r"fluid_blocks") / Path(r"fluid_1.vtk")

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
         
    def load_mesh_by_wind_velocity(self, wind_velocity: np.ndarray):
        wind_velocity_selection = get_wind_velocity_folder_name(wind_velocity)
        vtk_path = get_file_path(self.base_path / Path(wind_velocity_selection),  self.tail_path)
        mesh = pv.read(vtk_path)
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
                warnings.warn(
                    f"Point {pt} is outside the mesh domain; "
                    f"returned velocity is zero-padded."
                )
        
        return velocity

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

def get_wind_velocity_folder_name(wind_velocity: np.ndarray) -> str:
    """Generate a folder name from a 3D wind velocity vector."""
    wind_velocity = np.asarray(wind_velocity)
    
    if wind_velocity.shape != (3,):
        raise ValueError(f"wind_velocity must have shape (3,), got {wind_velocity.shape}")
    
    def make_sign_symbol(value):
        return "n" if value < 0 else "p"
    
    components = ["x", "y", "z"]
    parts = []
    
    for value, component in zip(wind_velocity, components):
        sign = make_sign_symbol(value)
        mag = abs(value)
        
        # Clean up float representation: 1.0 → "1", 0.5 → "0.5"
        if isinstance(mag, (float, np.floating)):
            if mag.is_integer():
                mag_str = str(int(mag))
            else:
                mag_str = f"{mag:.6g}"  # avoids 1.0000000000000002 noise
        else:
            mag_str = str(mag)
            
        parts.append(f"{component}{sign}{mag_str}")
    
    return "_".join(parts)
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

import numpy as np
import pyvista as pv


def plot_slices(
    mesh,
    y_offset=(0, 0.5, 0),
    z_offset=(0, 0, 0.5),
    arrow_factor: float = 0.05,
    x_stride: int = 5,
    y_stride: int = 5,
    z_stride: int = 5,
):
    """
    Plot y- and z-normal slices with subsampled velocity arrows.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh containing ``velocity`` point data.
    y_offset : array-like
        Offset from mesh center for the y-normal slice.
    z_offset : array-like
        Offset from mesh center for the z-normal slice.
    arrow_factor : float
        Arrow length scale.
    x_stride, y_stride, z_stride : int
        Sampling strides along each coordinate direction.
    """
    if "velocity" not in mesh.point_data:
        raise KeyError("'velocity' not found in mesh.point_data")

    if "speed" not in mesh.point_data:
        velocity = np.asarray(mesh.point_data["velocity"])
        mesh.point_data["speed"] = np.linalg.norm(velocity, axis=1)

    center = np.asarray(mesh.center)

    slice_y = mesh.slice(
        normal=(0, 1, 0),
        origin=center + np.asarray(y_offset, dtype=float),
    )

    slice_z = mesh.slice(
        normal=(0, 0, 1),
        origin=center + np.asarray(z_offset, dtype=float),
    )

    def make_sparse_arrows(
        slice_mesh,
        first_axis: int,
        second_axis: int,
        first_stride: int,
        second_stride: int,
    ):
        points = np.asarray(slice_mesh.points)
        vectors = np.asarray(slice_mesh.point_data["velocity"])

        first_values = np.unique(points[:, first_axis])
        second_values = np.unique(points[:, second_axis])

        selected_first = first_values[::first_stride]
        selected_second = second_values[::second_stride]

        mask = (
            np.isin(points[:, first_axis], selected_first)
            & np.isin(points[:, second_axis], selected_second)
        )

        arrow_points = pv.PolyData(points[mask])
        arrow_points.point_data["velocity"] = vectors[mask]

        return arrow_points.glyph(
            orient="velocity",
            scale="velocity",
            factor=arrow_factor,
        )

    # y-normal slice lies in the x-z plane
    arrows_y = make_sparse_arrows(
        slice_y,
        first_axis=0,
        second_axis=2,
        first_stride=x_stride,
        second_stride=z_stride,
    )

    # z-normal slice lies in the x-y plane
    arrows_z = make_sparse_arrows(
        slice_z,
        first_axis=0,
        second_axis=1,
        first_stride=x_stride,
        second_stride=y_stride,
    )

    plotter = pv.Plotter(
        shape=(1, 2),
        window_size=(1000, 500),
    )

    plotter.subplot(0, 0)
    plotter.add_mesh(arrows_y, show_scalar_bar=False)
    plotter.add_text("Slice y", font_size=12)
    plotter.show_axes()
    plotter.show_grid()

    plotter.subplot(0, 1)
    plotter.add_mesh(arrows_z, show_scalar_bar=False)
    plotter.add_text("Slice z", font_size=12)
    plotter.show_axes()
    plotter.show_grid()

    plotter.link_views()

    return plotter

def plot_velocity_arrows(
    mesh,
    factor: float = 0.1,
    x_stride: int = 5,
    y_stride: int = 5,
    z_stride: int = 5,
):
    """Plot subsampled velocity arrows on a structured mesh."""
    nx, ny, nz = mesh.dimensions

    if nx * ny * nz != mesh.n_points:
        raise ValueError(
            "This function requires a structured mesh with valid dimensions."
        )

    # VTK point IDs: x changes fastest, followed by y, then z.
    point_ids = np.arange(mesh.n_points).reshape(
        (nz, ny, nx)
    )

    selected_ids = point_ids[
        ::z_stride,
        ::y_stride,
        ::x_stride,
    ].ravel()

    arrow_points = pv.PolyData(
        np.asarray(mesh.points)[selected_ids]
    )
    arrow_points.point_data["velocity"] = np.asarray(
        mesh.point_data["velocity"]
    )[selected_ids]

    arrows = arrow_points.glyph(
        orient="velocity",
        scale="velocity",
        factor=factor,
    )

    plotter = pv.Plotter(window_size=(800, 600))
    plotter.add_mesh(arrows)
    plotter.show_axes()
    plotter.show_grid()

    return plotter

def plot_velocity_along_x(
    mesh,
    y: float,
    z: float,
    resolution: int = 500,
    tolerance: float = 1e-8,
) -> None:
    """
    Plot interpolated velocity along an x-directed line and overlay the
    original mesh points located on that line.

    Parameters
    ----------
    mesh
        PyVista mesh containing a ``velocity`` point-data array.
    y
        Constant y-coordinate of the sampling line.
    z
        Constant z-coordinate of the sampling line.
    resolution
        Number of intervals used for interpolated line sampling.
    tolerance
        Absolute tolerance used to identify original mesh points on the line.
    """
    points = np.asarray(mesh.points)
    velocity = np.asarray(mesh.point_data["velocity"])
    speed = np.linalg.norm(velocity, axis=1)

    # Make speed available to PyVista sampling.
    mesh.point_data["speed"] = speed

    # Interpolate along the x-axis at fixed y and z.
    x_min, x_max, _, _, _, _ = mesh.bounds

    sampled_line = mesh.sample_over_line(
        pointa=(x_min, y, z),
        pointb=(x_max, y, z),
        resolution=resolution,
    )

    x_sampled = np.asarray(sampled_line.points)[:, 0]
    velocity_sampled = np.asarray(
        sampled_line.point_data["velocity"]
    )
    speed_sampled = np.asarray(
        sampled_line.point_data["speed"]
    )

    # Find original mesh points lying on the same line.
    point_mask = (
        np.isclose(points[:, 1], y, atol=tolerance)
        & np.isclose(points[:, 2], z, atol=tolerance)
    )

    x_mesh = points[point_mask, 0]
    speed_mesh = speed[point_mask]

    order = np.argsort(x_mesh)
    x_mesh = x_mesh[order]
    speed_mesh = speed_mesh[order]

    # Plot interpolated values and original mesh points.
    plt.figure()

    plt.plot(
        x_sampled,
        velocity_sampled[:, 0],
        label="vx",
    )
    plt.plot(
        x_sampled,
        velocity_sampled[:, 1],
        label="vy",
    )
    plt.plot(
        x_sampled,
        velocity_sampled[:, 2],
        label="vz",
    )
    plt.plot(
        x_sampled,
        speed_sampled,
        label="Interpolated speed",
        linewidth=2,
    )
    plt.scatter(
        x_mesh,
        speed_mesh,
        s=20,
        label="Original mesh-point speed",
    )

    plt.xlabel("x")
    plt.ylabel("Velocity")
    plt.title(f"Velocity along y = {y}, z = {z}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
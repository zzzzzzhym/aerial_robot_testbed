import mujoco
import numpy as np


def setup_camera():
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0, 0, 0.5]
    cam.distance = 4.0
    cam.azimuth = 90.0
    cam.elevation = -20.0
    return cam


def setup_camera_inclined():
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0, 0, 0.4]
    cam.distance = 4.5
    cam.azimuth = 90.0
    cam.elevation = -15.0
    return cam


class MujocoWorld:
    def __init__(self, xml_path, render_height=480, render_width=640, camera=None):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        self.cam = camera if camera is not None else setup_camera()

    def simulate(self, duration=2.0, render_fps=30):
        mujoco.mj_resetData(self.model, self.data)

        body_id = self.model.body("box").id
        dof_adr = self.model.jnt_dofadr[self.model.joint("box_joint").id]
        mass = self.model.body_mass[body_id]

        frames = []
        times, positions, forces, accels = [], [], [], []
        frame_interval = 1.0 / render_fps
        next_frame_time = 0.0

        while self.data.time < duration:
            mujoco.mj_step(self.model, self.data)

            times.append(self.data.time)
            positions.append(self.data.xpos[body_id].copy())
            acc = self.data.qacc[dof_adr:dof_adr + 3].copy()
            accels.append(acc)
            forces.append(mass * acc)

            if self.data.time >= next_frame_time:
                self.renderer.update_scene(self.data, camera=self.cam)
                frames.append(self.renderer.render().copy())
                next_frame_time += frame_interval

        log = {
            "time":  np.array(times),
            "pos":   np.array(positions),
            "force": np.array(forces),
            "accel": np.array(accels),
        }
        return frames, log

    def show_animation(self, frames, fps=30, title=""):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from IPython.display import HTML, display

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        if title:
            ax.set_title(title)
        img_plot = ax.imshow(frames[0])

        def update(i):
            img_plot.set_array(frames[i])
            return [img_plot]

        ani = animation.FuncAnimation(
            fig, update, frames=len(frames), interval=1000 / fps, blit=True
        )
        plt.close(fig)
        display(HTML(ani.to_jshtml()))

    def plot_data(self, log, title=""):
        import matplotlib.pyplot as plt
        from IPython.display import display

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        if title:
            fig.suptitle(title)

        t = log["time"]

        ax = axes[0]
        for i, label in enumerate(["x", "y", "z"]):
            ax.plot(t, log["pos"][:, i], label=label)
        ax.set_ylabel("Position (m)")
        ax.set_title("Box center position — inertial frame")
        ax.legend()
        ax.grid(True)

        ax = axes[1]
        for i, label in enumerate(["Fx", "Fy", "Fz"]):
            ax.plot(t, log["force"][:, i], label=label)
        ax.set_ylabel("Force (N)")
        ax.set_title("Net force on box — inertial frame  (F = m·a)")
        ax.legend()
        ax.grid(True)

        ax = axes[2]
        for i, label in enumerate(["ax", "ay", "az"]):
            ax.plot(t, log["accel"][:, i], label=label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s²)")
        ax.set_title("Acceleration of box — inertial frame")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        display(fig)
        plt.close(fig)

    def close(self):
        self.renderer.close()

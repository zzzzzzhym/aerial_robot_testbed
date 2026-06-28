from pathlib import Path
from mujoco_world import MujocoWorld, setup_camera, setup_camera_inclined

XML_FLAT = Path(__file__).parent / "falling_box.xml"
XML_INCLINED = Path(__file__).parent / "inclined_box.xml"

if __name__ == "__main__":
    print("=== Scenario 1: 1kg box falling on flat floor ===")
    world1 = MujocoWorld(XML_FLAT, camera=setup_camera())
    frames1, log1 = world1.simulate(duration=2.0, render_fps=30)
    print(f"  {len(frames1)} frames rendered")
    world1.show_animation(frames1, title="Scenario 1: Flat Floor (falling box)")
    world1.plot_data(log1, title="Scenario 1: Flat Floor")
    world1.close()

    print("=== Scenario 2: 1kg box sliding on 50deg inclined floor (mu=1) ===")
    world2 = MujocoWorld(XML_INCLINED, camera=setup_camera_inclined())
    frames2, log2 = world2.simulate(duration=3.0, render_fps=30)
    print(f"  {len(frames2)} frames rendered")
    world2.show_animation(frames2, title="Scenario 2: Inclined Floor (sliding box, 50deg, mu=1)")
    world2.plot_data(log2, title="Scenario 2: Inclined Floor (50deg, mu=1)")
    world2.close()

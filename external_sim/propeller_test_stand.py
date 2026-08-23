import numpy as np

from external_sim.main_process import P600


class PropellerTestStand(P600):
    """Single-rotor test stand scenario.

    Identical to P600 except throttle assignment: rotor 0 ramps 0 → 1 over
    STAGE_1_DURATION seconds after an initial drop phase. Rotors 1-3 stay at 0.
    The controller command is ignored.

    Stage 0 (STAGE_0_DURATION sec): all throttles = 0, body drops onto floor.
    Stage 1 (STAGE_1_DURATION sec): rotor 0 throttle ramps linearly 0 → 1.
    """

    STAGE_0_DURATION = 1.0    # seconds
    STAGE_1_DURATION = 10.0   # seconds

    def set_motor_throttles(self, speeds: np.ndarray) -> dict:
        t = self.i * self.dt
        if t < self.STAGE_0_DURATION:
            throttle_0 = 0.0
        else:
            throttle_0 = float(np.clip(
                (t - self.STAGE_0_DURATION) / self.STAGE_1_DURATION, 0.0, 1.0
            ))
        return {
            setter: (throttle_0 if idx == 0 else 0.0)
            for idx, setter in enumerate(self.motor_throttle_setters)
        }

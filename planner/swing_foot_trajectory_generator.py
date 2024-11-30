import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import PiecewisePolynomial

class SwingFootTrajectoryGenerator():

    def __init__(self):
        # Initial and final foot positions in the world frame
        self.footpos_init = np.zeros((3, 1), dtype=np.float32)
        self.footpos_final = np.zeros((3, 1), dtype=np.float32)

    def set_init_foot_position(self, footpos_init):
        self.footpos_init = footpos_init

    def set_final_foot_position(self, footpos_final):
        self.footpos_final = footpos_final

    """
    Generate the swing foot trajectory at the current swing time.

    Args:
        swing_height: Maximum height on the swing trajectory
        total_swing_time: Total swing time
        cur_swing_time: Current swing time

    Returns:
        pos_swingfoot: Desired position of the swing foot in the world frame
        vel_swingfoot: Desired velocity of the swing foot in the world frame
        acc_swingfoot: Desired acceleration of the swing foot in the world frame
    """
    def generate_swing_foot_trajectory(
            self,
            swing_height,
            total_swing_time, 
            cur_swing_time,
        ):
        break_points = np.array([[0.],
                                 [total_swing_time / 2.0],
                                 [total_swing_time]], dtype=np.float32)

        footpos_middle_time = (self.footpos_init + self.footpos_final) / 2.0
        footpos_middle_time[2] = swing_height

        footpos_break_points = np.hstack((
            self.footpos_init.reshape(3, 1),
            footpos_middle_time.reshape(3, 1),
            self.footpos_final.reshape(3, 1)
        ))

        vel_break_points = np.zeros((3, 3), dtype=np.float32)

        swing_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            break_points, footpos_break_points, vel_break_points[:, 0], vel_break_points[:, 1]
        )

        pos_swingfoot = swing_traj.value(cur_swing_time)
        vel_swingfoot = swing_traj.derivative(1).value(cur_swing_time)
        acc_swingfoot = swing_traj.derivative(2).value(cur_swing_time)

        return np.squeeze(pos_swingfoot), np.squeeze(vel_swingfoot), np.squeeze(acc_swingfoot)


    """
    Compute the final foot position in the world frame for this leg's
    current swing.
    Assumes that the current leg is in swing phase.
    Call this once in the beginning of the swing phase.
    """
    def set_foot_placement(
        self,
        pos_body_world,     # Current position of the body in the world frame
        vel_body_world,     # Current velocity of the body in the world frame
        yaw_turn_rate,      # Desired yaw turn rate
        hip_position_body,  # Position of the hip (associated with this leg) in the body frame
        total_stance_time,  # Total stance time
    ):
        # Compute the hip offset
        hip_offset = hip_position_body

        # Compute the twisting vector
        twisting_vector = np.array([-hip_offset[1], hip_offset[0], 0.])

        # Compute the hip horizontal velocity
        hip_horizontal_velocity = vel_body_world + yaw_turn_rate * twisting_vector

        # Compute the foot target position
        foot_target_position = pos_body_world + hip_offset + \
                                hip_horizontal_velocity * total_stance_time
        foot_target_position[2] = 0.

        # Set the foot target position
        self.footpos_final = foot_target_position


def test_swing_foot_traj():
    # Initialize the swing foot trajectory generator
    swing_foot_traj_gen = SwingFootTrajectoryGenerator()

    # Set the initial foot position
    footpos_init = np.array([-0.176, 0.11, 0.], dtype=np.float32)
    swing_foot_traj_gen.set_init_foot_position(footpos_init)

    # Set the final foot position
    total_segments = 64
    stance_time = 8e-3 * total_segments/2
    swing_foot_traj_gen.set_foot_placement(
        np.array([0., 0., 0.], dtype=np.float32),
        np.array([0.1, 0., 0.], dtype=np.float32),
        0.0,
        np.array([-0.176, 0.11, 0.], dtype=np.float32),
        stance_time
    )

    # Generate trajectory for swing time from 0 to 1, and plot it
    swing_height = 0.1
    total_swing_time = 8e-3 * total_segments/2
    cur_swing_time = 0.0
    generated_traj = []
    phase = 0.0
    while phase < 1.0:
        pos_swingfoot, _, _ = swing_foot_traj_gen.generate_swing_foot_trajectory(
            swing_height,
            total_swing_time,
            phase * total_swing_time
        )
        print(pos_swingfoot)
        generated_traj.append(pos_swingfoot)
        phase += 2/total_segments

    generated_traj = np.array(generated_traj)
    print(generated_traj[:, 0])
    plt.plot(generated_traj[:, 0], generated_traj[:, 2])
    plt.show()

if __name__ == '__main__':
    test_swing_foot_traj()

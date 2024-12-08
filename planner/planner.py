from math import floor
import numpy as np
from pydrake.all import *
from planner.gait import Gait
from planner.swing_foot_trajectory_generator import SwingFootTrajectoryGenerator


class Planner(LeafSystem):
    """
    Implements a simple planner for the trunk model,
    it generates the desired positions, velocities and accelerations
    for the feet, center of mass and body orientation.

    For each plan, it only uses the current time from the context.

    It has two output ports:
    - Trunk trajectory (Vector valued): Desired positions, velocities and accelerations as a dictionary
    - Trunk geometry (Vector valued): To set the geometry of the trunk in the scene graph for visualization
    """
    def __init__(
            self,
            trunk_frame_ids: dict,
            mode: int = 0,
            horizon_length: int = 10,
            dt: float = 5e-3,
            gait: Gait = Gait.TROTTING160,
            foot_clearance: float = 0.0,
        ):
        LeafSystem.__init__(self)

        self.trunk_frame_ids_dict = trunk_frame_ids
        self.mode = mode
        self.horizon_length = horizon_length
        self.dt = dt
        self.foot_clearance = foot_clearance

        # For geometry, we need a FramePoseVector
        fpv = FramePoseVector()
        for _, frame_id in trunk_frame_ids.items():
            fpv.set_value(frame_id, RigidTransform())

        # Output ports
        self.output_ports = {
            "trunk_trajectory": self.DeclareAbstractOutputPort(
                "trunk_trajectory",
                lambda: AbstractValue.Make([]),
                self.CalcTrunkTrajectory
            ).get_index(),

            "trunk_geometry": self.DeclareAbstractOutputPort(
                "trunk_geometry",
                lambda: AbstractValue.Make(fpv),
                self.CalcTrunkGeometry
            ).get_index()
        }

        # List of trajectories, one for each time step till the horizon
        self.horizon_trajectory = []
        self.last_ran = -1.0

        # Gait object and iteration number (for the gait)
        self.gait = gait
        self.iteration_num = 0

        # Swing foot trajectory generator for each leg
        self.swing_foot_trajectory_generators = [
            SwingFootTrajectoryGenerator()
            for _ in range(4)
        ]

        # Calculate the position of hips in body frame for each leg
        # Same as standing plan
        self.hip_positions_body = [
            np.array([0.176, 0.11, 0.0]),
            np.array([0.176, -0.11, 0.0]),
            np.array([-0.20, 0.11, 0.0]),
            np.array([-0.20, -0.11, 0.0])
        ]


    def get_output_port_by_name(self, name):
        """
        Get the output port object by name
        """
        return self.get_output_port(self.output_ports[name])


    def StandingPlan(self, t):
        """
        Generate a standing plan
        """
        output_trajectory = {
            "p_lf": np.array([0.176,   0.11, self.foot_clearance]), # Left front foot
            "p_rf": np.array([0.176,  -0.11, self.foot_clearance]), # Right front foot
            "p_lh": np.array([-0.203,  0.11, self.foot_clearance]), # Left hind foot
            "p_rh": np.array([-0.203, -0.11, self.foot_clearance]), # Right hind foot

            "v_lf": np.array([0.0, 0.0, 0.0]), # Left front foot
            "v_rf": np.array([0.0, 0.0, 0.0]), # Right front foot
            "v_lh": np.array([0.0, 0.0, 0.0]), # Left hind foot
            "v_rh": np.array([0.0, 0.0, 0.0]), # Right hind foot

            "a_lf": np.array([0.0, 0.0, 0.0]), # Left front foot
            "a_rf": np.array([0.0, 0.0, 0.0]), # Right front foot
            "a_lh": np.array([0.0, 0.0, 0.0]), # Left hind foot
            "a_rh": np.array([0.0, 0.0, 0.0]), # Right hind foot

            "contact_states": np.array([True, True, True, True]), # Contact states for each foot

            "p_com": np.array([0.0, 0.0, 0.3]), # Center of mass
            "v_com": np.array([0.0, 0.0, 0.0]), # Center of mass
            "a_com": np.array([0.0, 0.0, 0.0]), # Center of mass

            "rpy": np.array([0.0, 0.0, 0.0]), # Roll, pitch, yaw
            "rpy_dot": np.array([0.0, 0.0, 0.0]), # Roll, pitch, yaw derivatives
            "rpy_ddot": np.array([0.0, 0.0, 0.0]) # Roll, pitch, yaw second derivatives
        }
        return output_trajectory

    
    def TurningHeadPlan(self, t):
        """
        Generate a plan for turning the head
        """
        output_trajectory = self.StandingPlan(t)
        output_trajectory["rpy"] = np.array([0.0, 0.0, 0.1 * np.sin(t)])
        output_trajectory["rpy_dot"] = np.array([0.0, 0.0, 0.1 * np.cos(t)])

        return output_trajectory


    def RaiseFoot(self, t):
        """
        Modify the simple standing output values to lift one foot
        off the ground.
        """
        output_trajectory = self.StandingPlan(t)
        # output_trajectory["p_com"] += np.array([-0.01, 0.01, 0.0])

        a_foot = 3.0
        floor_t = floor(t)
        frac_t = t - floor_t
        rise_time = 0.1
        start_time = 1.0
        if t >= start_time and frac_t < rise_time:
            # Lift the right front foot
            output_trajectory["a_rf"] += np.array([ 0.0, 0.0, a_foot])
            output_trajectory["v_rf"] += np.array([ 0.0, 0.0, a_foot * (frac_t)])
            output_trajectory["contact_states"] = [True,False,True,True]
            output_trajectory["p_rf"] += np.array([ 0.0, 0.0, self.foot_clearance + 0.5 * a_foot * (frac_t)**2])

            # Also lift the left hind foot
            # output_trajectory["a_lh"] += np.array([ 0.0, 0.0, a_foot])
            # output_trajectory["v_lh"] += np.array([ 0.0, 0.0, a_foot * (frac_t)])
            # output_trajectory["contact_states"] = [True,False,False,True]
            # output_trajectory["p_lh"] += np.array([ 0.0, 0.0, self.foot_clearance + 0.5 * a_foot * (frac_t)**2])

        # Let it rest at that position for rise_time duration
        elif t >= start_time and frac_t >= rise_time and frac_t < 2 * rise_time:
            dist_start = self.foot_clearance + 0.5 * a_foot * (rise_time**2)
            output_trajectory["a_rf"] = np.array([0.0, 0.0, 0.0])
            output_trajectory["v_rf"] = np.array([0.0, 0.0, 0.0])
            output_trajectory["p_rf"] += np.array([0.0, 0.0, dist_start])
            output_trajectory["contact_states"] = [True,False,True,True]
            
            # output_trajectory["a_lh"] = np.array([0.0, 0.0, 0.0])
            # output_trajectory["v_lh"] = np.array([0.0, 0.0, 0.0])
            # output_trajectory["p_lh"] += np.array([0.0, 0.0, dist_start])
            # output_trajectory["contact_states"] = [True,False,False,True]

        elif t >= start_time and frac_t < 3 * rise_time:
            # Bring the foot back to the ground
            dist_start = self.foot_clearance + 0.5 * a_foot * (rise_time**2)
            output_trajectory["a_rf"] += np.array([ 0.0, 0.0, -a_foot])
            output_trajectory["v_rf"] += np.array([ 0.0, 0.0, -a_foot * (frac_t - 2*rise_time)])
            output_trajectory["contact_states"] = [True,False,True,True]
            output_trajectory["p_rf"] += np.array([ 0.0, 0.0, dist_start - 0.5 * a_foot * (frac_t - 2*rise_time)**2])

            # output_trajectory["a_lh"] += np.array([ 0.0, 0.0, -a_foot])
            # output_trajectory["v_lh"] += np.array([ 0.0, 0.0, -a_foot * (frac_t - rise_time)])
            # output_trajectory["contact_states"] = [True,False,False,True]
            # output_trajectory["p_lh"] += np.array([ 0.0, 0.0, dist_start - 0.5 * a_foot * (frac_t - rise_time)**2])

        elif t >= start_time and frac_t >= 3 * rise_time:
            output_trajectory["a_rf"] = np.array([0.0, 0.0, 0.0])
            output_trajectory["v_rf"] = np.array([0.0, 0.0, 0.0])
            
            # output_trajectory["a_lh"] = np.array([0.0, 0.0, 0.0])
            # output_trajectory["v_lh"] = np.array([0.0, 0.0, 0.0])
            output_trajectory["contact_states"] = [True,True,True,True]

        return output_trajectory
    

    def WalkingPlan(self, t):
        """
        The plan is to make the robot move with a desired velocity in the straight direction
        relative to the body frame. And also make the robot turn around the z-axis with a constant
        angular velocity - essentially a circular motion.
        """
        v_straight_des_body = 0.4  # Desired velocity in the straight direction relative to the body frame
        omega_des_z = 0.0  # Desired angular velocity around the z-axis in the world frame

        if t < 3.0:
            output_trajectory = self.StandingPlan(t)
            return output_trajectory
        
        # Update the iteration number in the gait
        self.gait.set_iteration(self.iteration_num)
        self.iteration_num += 1
        
        # Get the current plan
        previous_plan = self.horizon_trajectory[-1]

        # Update the plan
        output_trajectory = previous_plan.copy()

        # Find the transformation matrix from the body to the world frame
        rpy = previous_plan["rpy"]
        X_body_to_world = RigidTransform()
        X_body_to_world.set_translation(previous_plan["p_com"])
        X_body_to_world.set_rotation(RotationMatrix(RollPitchYaw(rpy)))

        # Find the velocity in the world frame
        v_body_des_world = X_body_to_world.rotation().multiply([v_straight_des_body, 0.0, 0.0])
        output_trajectory["v_com"] = v_body_des_world

        # Update the position of the center of mass
        pos_body_des_world = output_trajectory["p_com"] + v_body_des_world * self.dt
        output_trajectory["p_com"] = pos_body_des_world

        # Update the orientation of the body
        rpy[2] += omega_des_z * self.dt
        output_trajectory["rpy"] = rpy
        output_trajectory["rpy_dot"] = np.array([0.0, 0.0, omega_des_z])

        # Get the contact states from the gait
        gait_table = self.gait.get_gait_table()
        swing_table = self.gait.get_swing_state()
        output_trajectory["contact_states"] = gait_table

        # Now, we need to calculate the foot positions, velocities and accelerations
        # based on the gait table
        foot_names = ["lf", "rf", "lh", "rh"]
        for i, foot_name in enumerate(foot_names):
            if gait_table[i] == True:
                # If it switched from swing to stance
                if previous_plan["contact_states"][i] == False:
                    # Generate the end of previous swing trajectory
                    pos_swingfoot, vel_swingfoot, acc_swingfoot = self.swing_foot_trajectory_generators[i].generate_swing_foot_trajectory(
                        swing_height=0.1,
                        total_swing_time=self.gait.swing_time,
                        cur_swing_time=self.gait.swing_time
                    )

                    output_trajectory[f"p_{foot_name}"] = pos_swingfoot
                    output_trajectory[f"v_{foot_name}"] = vel_swingfoot
                    output_trajectory[f"a_{foot_name}"] = acc_swingfoot
                else:
                    # Stance phase - same position as before
                    output_trajectory[f"p_{foot_name}"] = previous_plan[f"p_{foot_name}"]
                    output_trajectory[f"v_{foot_name}"] = np.array([0.0, 0.0, 0.0])
                    output_trajectory[f"a_{foot_name}"] = np.array([0.0, 0.0, 0.0])
            else:
                # Swing phase
                # Check if its a new swing phase
                if previous_plan["contact_states"][i] == True:
                    # Set the initial position of the foot
                    self.swing_foot_trajectory_generators[i].set_init_foot_position(
                        previous_plan[f"p_{foot_name}"]
                    )

                    # Compute the final foot position
                    self.swing_foot_trajectory_generators[i].set_foot_placement(
                        pos_body_des_world,
                        v_body_des_world,
                        omega_des_z,
                        self.hip_positions_body[i],
                        self.gait.stance_time,
                    )
                    

                # Continue the swing phase
                curr_swing_time = swing_table[i] * self.gait.swing_time

                # Generate the swing foot trajectory
                pos_swingfoot, vel_swingfoot, acc_swingfoot = self.swing_foot_trajectory_generators[i].generate_swing_foot_trajectory(
                    swing_height=0.1,
                    total_swing_time=self.gait.swing_time,
                    cur_swing_time=curr_swing_time
                )

                # Update the output trajectory
                output_trajectory[f"p_{foot_name}"] = pos_swingfoot
                output_trajectory[f"v_{foot_name}"] = vel_swingfoot
                output_trajectory[f"a_{foot_name}"] = acc_swingfoot

        return output_trajectory



    def set_output_trajectory(self, t, mode):
        """
        Set the output trajectory based on the mode
        """
        # This is to avoid recomputing the same plan
        if self.last_ran == t:
            return

        # Choose the function based on the mode
        if mode == 0:
            trajectory_fxn = self.StandingPlan
        elif mode == 1:
            trajectory_fxn = self.TurningHeadPlan
        elif mode == 2:
            trajectory_fxn = self.RaiseFoot
        elif mode == 3:
            trajectory_fxn = self.WalkingPlan
        else:
            raise NotImplementedError("Mode not implemented")

        # Generate the plan based on the mode
        curr_len = len(self.horizon_trajectory)
        if curr_len == 0:
            for i in range(self.horizon_length):
                self.horizon_trajectory.append(trajectory_fxn(t + i * self.dt))
        else:
            self.horizon_trajectory.append(trajectory_fxn(t + self.horizon_length * self.dt))
            self.horizon_trajectory.pop(0)
        
        # Update the last ran time
        self.last_ran = t


    def CalcTrunkTrajectory(self, context, output):
        """
        Calculate the trunk trajectory
        """
        t = context.get_time()

        # Set the output trajectory
        self.set_output_trajectory(t, self.mode)

        # Set the output to the port
        output.set_value(self.horizon_trajectory)

    
    def CalcTrunkGeometry(self, context, output):
        """
        Calculate the trunk geometry
        """
        # Get the current time
        t = context.get_time()

        # Set the output trajectory
        self.set_output_trajectory(t, self.mode)

        # Current output trajectory
        output_trajectory = self.horizon_trajectory[0]

        # Create a FramePoseVector
        fpv = FramePoseVector()
        
        # Set the trunk frame
        trunk_transform = RigidTransform()
        trunk_transform.set_translation(output_trajectory["p_com"])
        trunk_transform.set_rotation(RollPitchYaw(output_trajectory["rpy"]).ToRotationMatrix())
        fpv.set_value(self.trunk_frame_ids_dict["trunk"], trunk_transform)

        # Set the foot frames
        for frame_name in ["lf", "rf", "lh", "rh"]:
            frame_id = self.trunk_frame_ids_dict[frame_name]
            foot_transform = RigidTransform()
            foot_transform.set_translation(output_trajectory[f"p_{frame_name}"])
            fpv.set_value(frame_id, foot_transform)

        # Set the output to the port
        output.set_value(fpv)
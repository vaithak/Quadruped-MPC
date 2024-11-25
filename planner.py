import numpy as np
from pydrake.all import *


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
    def __init__(self, trunk_frame_ids: dict, mode: int = 0):
        LeafSystem.__init__(self)

        self.trunk_frame_ids_dict = trunk_frame_ids
        self.mode = mode

        # For geometry, we need a FramePoseVector
        fpv = FramePoseVector()
        for _, frame_id in trunk_frame_ids.items():
            fpv.set_value(frame_id, RigidTransform())

        # Output ports
        self.output_ports = {
            "trunk_trajectory": self.DeclareAbstractOutputPort(
                "trunk_trajectory",
                lambda: AbstractValue.Make({}),
                self.CalcTrunkTrajectory
            ).get_index(),

            "trunk_geometry": self.DeclareAbstractOutputPort(
                "trunk_geometry",
                lambda: AbstractValue.Make(fpv),
                self.CalcTrunkGeometry
            ).get_index()
        }

        # Output dictionary - will be used by both output ports
        self.output_trajectory = {}
        self.last_ran = -1.0


    def get_output_port_by_name(self, name):
        """
        Get the output port object by name
        """
        return self.get_output_port(self.output_ports[name])


    def StandingPlan(self, t):
        """
        Generate a standing plan
        """
        self.output_trajectory = {
            "p_lf": np.array([0.175,  0.11, 0.0]), # Left front foot
            "p_rf": np.array([0.175, -0.11, 0.0]), # Right front foot
            "p_lh": np.array([-0.2,   0.11, 0.0]), # Left hind foot
            "p_rh": np.array([-0.2,  -0.11, 0.0]), # Right hind foot

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
        }


    def set_output_trajectory(self, t, mode):
        """
        Set the output trajectory based on the mode
        """
        # This is to avoid recomputing the same plan
        if self.last_ran == t:
            return
        
        if mode == 0:
            self.StandingPlan(t)
        else:
            raise NotImplementedError("Mode not implemented")
        
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
        output.set_value(self.output_trajectory)

    
    def CalcTrunkGeometry(self, context, output):
        """
        Calculate the trunk geometry
        """
        # Get the current time
        t = context.get_time()

        # Set the output trajectory
        self.set_output_trajectory(t, self.mode)

        # Create a FramePoseVector
        fpv = FramePoseVector()
        
        # Set the trunk frame
        trunk_transform = RigidTransform()
        trunk_transform.set_translation(self.output_trajectory["p_com"])
        trunk_transform.set_rotation(RollPitchYaw(self.output_trajectory["rpy"]).ToRotationMatrix())
        fpv.set_value(self.trunk_frame_ids_dict["trunk"], trunk_transform)

        # Set the foot frames
        for frame_name in ["lf", "rf", "lh", "rh"]:
            frame_id = self.trunk_frame_ids_dict[frame_name]
            foot_transform = RigidTransform()
            foot_transform.set_translation(self.output_trajectory[f"p_{frame_name}"])
            fpv.set_value(frame_id, foot_transform)

        # Set the output to the port
        output.set_value(fpv)
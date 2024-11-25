import numpy as np
from pydrake.all import *

class Controller(LeafSystem):
    """
    Implementing the MPC and swing leg controller for the trunk model.
    
    Based on the paper: 
    Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control.
    https://dspace.mit.edu/bitstream/handle/1721.1/138000/convex_mpc_2fix.pdf

    The controller has one input port and one output port:
    - Quadruped state: Input port to receive the state of the quadruped model

    Output port:
    - Quadruped torques: The torques to be applied to the quadruped model
    """

    def __init__(self, plant, dt):
        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.dt = dt

        # Quadruped state and trunk trajectory input ports
        self.input_ports = {
            "quadruped_state": self.DeclareVectorInputPort(
                "quadruped_state",
                BasicVector(self.plant.num_positions() + self.plant.num_velocities())
            ).get_index(),

            "trunk_input": self.DeclareAbstractInputPort(
                "trunk_input",
                AbstractValue.Make({})
            ).get_index()
        }

        # Quadruped torques output port
        self.output_ports = {
            "quadruped_torques": self.DeclareVectorOutputPort(
                "quadruped_torques",
                BasicVector(self.plant.num_actuators()),
                self.CalcQuadrupedTorques
            ).get_index()
        }


    def get_output_port_by_name(self, name):
        """
        Get the output port object by name
        """
        return self.get_output_port(self.output_ports[name])


    def get_input_port_by_name(self, name):
        """
        Get the input port object by name
        """
        return self.get_input_port(self.input_ports[name])


    def UpdateStoredContext(self, context):
        """
        Update the stored context with the current context
        """
        state = self.EvalVectorInput(context, self.input_ports["quadruped_state"]).get_value()
        self.plant.SetPositionsAndVelocities(self.plant_context, state)
        

    def CalcQuadrupedTorques(self, context, output):
        """
        Calculate the torques to be applied to the quadruped model
        """
        # Update the stored context
        self.UpdateStoredContext(context)
        q = self.plant.GetPositions(self.plant_context)
        v = self.plant.GetVelocities(self.plant_context)

        # Get the trunk trajectory
        trunk_trajectory = self.EvalAbstractInput(context, self.input_ports["trunk_input"]).get_value()

        # Calculate the torques
        torques = self.ControlLaw(q, v, trunk_trajectory)

        # Set the torques
        output.SetFromVector(torques)


    def ControlLaw(self, q, v, trunk_trajectory):
        """
        Implement the control law
        """
        # Get the trunk trajectory
        p_lf = trunk_trajectory["p_lf"]
        p_rf = trunk_trajectory["p_rf"]
        p_lh = trunk_trajectory["p_lh"]
        p_rh = trunk_trajectory["p_rh"]

        # Contact of feet
        contact_vector = trunk_trajectory["contact_states"]

        # Calculate the torques
        torques = np.zeros(self.plant.num_actuators())
        return torques
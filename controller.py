import numpy as np
from pydrake.all import *

DEBUG = True

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

    def __init__(
            self, 
            plant, 
            dt,
            mpc_horizon_length, # In number of discrete steps
            gravity_value, # Gravity value
            mu, # Friction coefficient
        ):
        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.dt = dt
        self.mpc_horizon_length = mpc_horizon_length
        self.gravity_value = gravity_value
        self.mu = mu
        self.world_frame = plant.world_frame()

        # Get the mass and inertia of the trunk
        self.mass = plant.CalcTotalMass(self.plant_context)
        I_moment = plant.GetBodyByName('body').CalcSpatialInertiaInBodyFrame(self.plant_context).CalcRotationalInertia().CopyToFullMatrix3()
        print("I_moment: \n", I_moment)
        print("mass: ", self.mass)
        self.I_moment_inv = np.linalg.inv(I_moment)

        # Get the actuation matrix
        self.B = plant.MakeActuationMatrix()

        # Linearized dynamics
        # States are: (for trunk rigid body only)
        # 1. 3D angular position
        # 2. 3D linear position
        # 3. 3D angular velocity
        # 4. 3D linear velocity
        # 5. Extra gravity term (1D)
        # This makes the state 13 dimensional
        self.mpc_state_dim = 13

        # Control inputs are:
        # 3D forces on each foot - 12 dimensional
        self.mpc_control_dim = 12

        # Q and R matrices for the cost function
        # r, p, y, x, y, z, omega_x, omega_y, omega_z, v_x, v_y, v_z, g
        self.Q = np.diag([5., 5., 50., 10., 50., 50., 1, 1, .1, 0.5, .1, .1, .0])
        self.R = 1e-6 * np.eye(self.mpc_control_dim)

        # Quadruped state and trunk trajectory input ports
        self.input_ports = {
            "quadruped_state": self.DeclareVectorInputPort(
                "quadruped_state",
                BasicVector(self.plant.num_positions() + self.plant.num_velocities())
            ).get_index(),

            "trunk_input": self.DeclareAbstractInputPort(
                "trunk_input",
                AbstractValue.Make([])
            ).get_index(),

            "contact_results": self.DeclareAbstractInputPort(
                "contact_results",
                AbstractValue.Make(ContactResults())
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

        # Store current state
        self.x_curr = np.zeros(self.mpc_state_dim)

        # Store current foot positions and velocities in world frame
        self.foot_positions = []
        self.foot_velocities = []

        # Kp, Kd for the swing controller
        self.Kp_swing = 200.0 * np.eye(3)
        self.Kd_swing = 20.0 * np.eye(3)

        # Store the contact results
        self.contact_results = ContactResults()

        # Store the torques
        self.torques = np.zeros(self.plant.num_actuators())


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


    def UpdateStoredContextAndState(self, context):
        """
        Update the stored context with the current context
        """
        state = self.EvalVectorInput(context, self.input_ports["quadruped_state"]).get_value()
        self.plant.SetPositionsAndVelocities(self.plant_context, state)

        # Get the contact results
        self.contact_results = self.EvalAbstractInput(context, self.input_ports["contact_results"]).get_value()

        # Get the current state
        curr_pose = self.plant.GetBodyByName("body").EvalPoseInWorld(self.plant_context)
        curr_vel  = self.plant.GetBodyByName("body").EvalSpatialVelocityInWorld(self.plant_context)
        self.x_curr[0:3]  = curr_pose.rotation().ToRollPitchYaw().vector()
        self.x_curr[3:6]  = curr_pose.translation()
        self.x_curr[6:9]  = curr_vel.rotational()
        self.x_curr[9:12] = curr_vel.translational()
        self.x_curr[12] = self.gravity_value

        if DEBUG:
            print("Current state: \n", np.round(self.x_curr, 2))

        # Get the foot positions
        self.foot_positions = [
            self.plant.GetBodyByName("LF_FOOT").EvalPoseInWorld(self.plant_context).translation(),
            self.plant.GetBodyByName("RF_FOOT").EvalPoseInWorld(self.plant_context).translation(),
            self.plant.GetBodyByName("LH_FOOT").EvalPoseInWorld(self.plant_context).translation(),
            self.plant.GetBodyByName("RH_FOOT").EvalPoseInWorld(self.plant_context).translation()
        ]

        # Get the foot velocities
        self.foot_velocities = [
            self.plant.GetBodyByName("LF_FOOT").EvalSpatialVelocityInWorld(self.plant_context).translational(),
            self.plant.GetBodyByName("RF_FOOT").EvalSpatialVelocityInWorld(self.plant_context).translational(),
            self.plant.GetBodyByName("LH_FOOT").EvalSpatialVelocityInWorld(self.plant_context).translational(),
            self.plant.GetBodyByName("RH_FOOT").EvalSpatialVelocityInWorld(self.plant_context).translational()
        ]
        

    def CalcQuadrupedTorques(self, context, output):
        """
        Calculate the torques to be applied to the quadruped model
        """
        # Update the stored context and state
        self.UpdateStoredContextAndState(context)

        # Get the trunk trajectory
        trunk_trajectory = self.EvalAbstractInput(context, self.input_ports["trunk_input"]).get_value()

        # Calculate the torques
        self.torques = self.ControlLaw(trunk_trajectory)

        # Set the torques
        output.SetFromVector(self.torques)


    def ControlLaw(self, trunk_trajectory):
        """
        Implement the control law
        """
        # Calculate contact forces for each foot in body frame
        forces = self.CalcForcesMPC(trunk_trajectory)
        if DEBUG:
            print("Forces: \n", np.round(forces, 2))

        # Get the contact states
        contact_states = trunk_trajectory[0]["contact_states"]

        # Calculate the torques because of the forces
        foot_Js = self.CalcFootTranslationJacobians()
        u = np.zeros(self.mpc_control_dim)
        for i in range(4):
            if contact_states[i]:
                u[i*3:i*3+3] = -foot_Js[i].T @ forces[i]
            else:
                u[i*3:i*3+3] = self.CalculateTorquesSwingController(trunk_trajectory)[i*3:i*3+3]

        # Use actuation matrix to get the torques
        torques = u

        return torques

    def CalcForcesMPC(self, trunk_trajectory):
        """
        Calculate the contact forces for each foot using Model Predictive Control
        """
        # Initialize mathematical program and decalre decision variables
        prog = MathematicalProgram()
        x = np.zeros((self.mpc_horizon_length, self.mpc_state_dim), dtype=Variable)
        for i in range(self.mpc_horizon_length):
            x[i] = prog.NewContinuousVariables(self.mpc_state_dim, "x_" + str(i))
        f = np.zeros((self.mpc_horizon_length-1, self.mpc_control_dim), dtype=Variable)
        for i in range(self.mpc_horizon_length-1):
            f[i] = prog.NewContinuousVariables(self.mpc_control_dim, "f_" + str(i))

        # Create x_i_ref for all time steps
        def GetReferenceState(trunk_trajectory, i, x_curr):
            """
            Get the reference state at time i
            # TODO: use i to get the reference state
            """
            x_ref = np.zeros(self.mpc_state_dim)
            x_ref[0:3] = trunk_trajectory[i]["rpy"]
            x_ref[3:6] = trunk_trajectory[i]["p_com"]
            x_ref[9:12] = trunk_trajectory[i]["v_com"]
            x_ref[12] = self.gravity_value

            # Convert from rpy_dot to omega
            rot_matrix = RollPitchYaw(x_ref[0:3]).ToRotationMatrix().inverse()
            x_ref[6:9] = rot_matrix @ trunk_trajectory[i]["rpy_dot"]

            # Compensation for pitch and roll drift
            x_ref[0] = 0.0 - x_curr[0]
            x_ref[1] = 0.0 - x_curr[1]

            return x_ref

        x_ref = np.zeros((self.mpc_horizon_length, self.mpc_state_dim))
        for i in range(self.mpc_horizon_length):
            x_ref[i] = GetReferenceState(trunk_trajectory, i, self.x_curr)
        
        ######### Add the dynamics constraints #########
        for i in range(self.mpc_horizon_length-1):
            A_i, B_i = self.DiscreteTimeLinearizedDynamics(trunk_trajectory, i)
            prog.AddLinearEqualityConstraint(x[i+1] - A_i @ x[i] - B_i @ f[i], np.zeros(self.mpc_state_dim))

        ######### Add the force constraints on non-contact feet #########
        # For creating a constraint D_i @ f_i = 0, to zero out the 
        # forces when the foot is not in contact
        for i in range(self.mpc_horizon_length-1):
            D_i = np.zeros((self.mpc_control_dim, self.mpc_control_dim))
            contact_states = trunk_trajectory[i]["contact_states"]
            for j in range(4):
                if not contact_states[j]:
                    D_i[j*3:j*3+3, j*3:j*3+3] = np.eye(3)
            prog.AddLinearEqualityConstraint(D_i @ f[i], np.zeros(self.mpc_control_dim))

        ######### Add the force limits on contact feet #########
        # force_z is within f_min and f_max, and the force_x and force_y
        # are within the friction cone
        f_min = 0
        f_max = 200
        for i in range(self.mpc_horizon_length-1):
            contact_states = trunk_trajectory[i]["contact_states"]
            for j in range(4):
                if contact_states[j]:
                    # f_z is within f_min and f_max
                    prog.AddBoundingBoxConstraint(f_min, f_max, f[i][j*3+2])

                    # f_x within the friction cone
                    prog.AddLinearConstraint(f[i][j*3] <= self.mu * f[i][j*3+2])
                    prog.AddLinearConstraint(-self.mu * f[i][j*3+2] <= f[i][j*3])

                    # f_y within the friction cone
                    prog.AddLinearConstraint(f[i][j*3+1] <= self.mu * f[i][j*3+2])
                    prog.AddLinearConstraint(-self.mu * f[i][j*3+2] <= f[i][j*3+1])

        ######### Add the initial state constraint #########
        # derived from the current state of the robot
        prog.AddLinearEqualityConstraint(x[0] - self.x_curr, np.zeros(self.mpc_state_dim))

        ######### Add the cost function #########
        for i in range(self.mpc_horizon_length):
            prog.AddQuadraticCost((x[i] - x_ref[i]).T @ self.Q @ (x[i] - x_ref[i]))
            if i < self.mpc_horizon_length-1:
                prog.AddQuadraticCost(f[i].T @ self.R @ f[i])

        ######### Solve the optimization problem #########
        solver = OsqpSolver()
        prog.SetSolverOption(solver.id(), "max_iter", 1000)

        result = solver.Solve(prog)

        forces = np.zeros(self.mpc_control_dim)
        if not result.is_success():
            raise RuntimeError("MPC failed to find a solution")
        else:
            forces = result.GetSolution(f[0])

        return forces.reshape((4, 3))


    """
    The swing controller computes and follows a trajectory
    for the foot in the world coordinate system. The controller
    for tracking the trajectory uses the sum of a feedback and
    feedforward term to compute joint torques.
    The control law used to compute joint torques for leg i is
    τi = Ji.T @ Kp @ (pi,ref_body_frame - pi_body_frame) + Kd(vi,ref_body_frame - vi_body_frame)] + τi,ff

    Where Ji is the Jacobian of the foot i in the body frame,
    pi,ref_body_frame is the desired position of the foot i in the body frame,
    pi_body_frame is the current position of the foot i in the body frame,
    vi,ref_body_frame is the desired velocity of the foot i in the body frame,
    vi_body_frame is the current velocity of the foot i in the body frame,
    Kp is the proportional gain positive definite diagonal matrix,
    Kd is the derivative gain positive definite diagonal matrix,
    τi,ff is the feedforward torque.

    The feedforward torque is computed as follows:
    τi,ff = Ji.T @ Λi(ai,ref_body_frame - Ji' @ qi') + Ci @ qi' + Gi

    Where Λi is the opertational space inertia matrix of the foot i in the body frame,
    ai,ref_body_frame is the desired acceleration of the foot i in the body frame,
    qi' is the 3d joint velocity of the foot i,
    Ci is the coriolis and centrifugal forces matrix for the foot i,
    Gi is the gravity forces matrix for the foot i.
    """
    def CalculateTorquesSwingController(self, trunk_trajectory):
        """
        Calculate the torques because of the swing controller
        """
        # Get the contact states
        contact_states = trunk_trajectory[0]["contact_states"]

        # Initialize the torques
        torques = np.zeros(self.mpc_control_dim)

        # Foot names
        foot_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
        short_names = ["lf", "rf", "lh", "rh"]

        # Calculate the torques for each foot
        for i in range(4):
            curr_foot = foot_names[i]
            if contact_states[i]:
                continue

            # Caclulate the desired position, velocity and acceleration
            p_des_world = trunk_trajectory[0]["p_" + short_names[i]]
            v_des_world = trunk_trajectory[0]["v_" + short_names[i]]
            a_des_world = trunk_trajectory[0]["a_" + short_names[i]]
            p_curr_world = self.foot_positions[i]
            v_curr_world = self.foot_velocities[i]

            # Calculate the Jacobian
            J = self.plant.CalcJacobianTranslationalVelocity(
                self.plant_context,
                JacobianWrtVariable.kV,
                self.plant.GetFrameByName(curr_foot),
                np.zeros(3),
                self.plant.world_frame(),
                self.plant.world_frame()
            )
            J_leg = J[:, 6 + i*3:6 + i*3 + 3]

            # tau_fb
            tau_fb = self.Kp_swing @ (p_des_world - p_curr_world) + \
                        self.Kd_swing @ (v_des_world - v_curr_world)

            tau_fb = J_leg.T @ tau_fb
            
            # Calculate J.T @ Operational Space Inertia Matrix
            M = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)[6 + i*3:6 + i*3 + 3, 6 + i*3:6 + i*3 + 3]
            J_T_Delta_i = M @ np.linalg.inv(J_leg)

            # Calculate the coriolis and gravity terms only for the current leg
            Cv_i = self.plant.CalcBiasTerm(self.plant_context)[6 + i*3:6 + i*3 + 3]
            Gv_i = -self.plant.CalcGravityGeneralizedForces(self.plant_context)[6 + i*3:6 + i*3 + 3]

            # Jdot_qi_dot
            Jdot_qi_dot = self.plant.CalcBiasTranslationalAcceleration(
                self.plant_context,
                JacobianWrtVariable.kV,
                self.plant.GetFrameByName(curr_foot),
                np.zeros(3),
                self.plant.world_frame(),
                self.plant.world_frame()
            ).squeeze()

            # tau_ff
            tau_ff = np.zeros(3)
            tau_ff = J_T_Delta_i @ (a_des_world - Jdot_qi_dot) + Cv_i + Gv_i

            # Set the desired torque
            torques[i*3:i*3+3] = tau_fb + tau_ff

        return torques


    def CalcFootTranslationJacobians(self):
        """
        Calculate the foot translation Jacobians
        """
        # Get the foot positions
        foot_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]

        # Calculate the foot translation Jacobians
        J = []
        i = 0
        for foot in foot_names:
            # Calculate the Jacobian
            curr_J = self.plant.CalcJacobianTranslationalVelocity(
                self.plant_context,
                JacobianWrtVariable.kV,
                self.plant.GetFrameByName(foot),
                np.zeros(3),
                self.world_frame,
                self.world_frame
            )
            curr_J = curr_J[:, 6 + i*3:6 + i*3 + 3]
            J.append(curr_J)
            i += 1
        return J
    

    def ContinuosTimeLinearizedDynamics(self, trunk_trajectory, horizon_idx):
        """
        Calculate the continuous time linearized dynamics at horizon_idx of the trajectory
        """
        

        A = np.zeros((self.mpc_state_dim, self.mpc_state_dim))
        B = np.zeros((self.mpc_state_dim, self.mpc_control_dim))

        ######### Fill in the A matrix #########
        # Linear Velocity Dynamics
        A[3:6, self.mpc_state_dim-4:self.mpc_state_dim-1] = np.eye(3)

        # Linear Acceleration Dynamics - just gravity
        A[9:12, self.mpc_state_dim-1] = np.array([0, 0, -1])

        # Angular Velocity Dynamics
        R = RollPitchYaw(self.x_curr[0:3]).ToRotationMatrix().matrix()
        
        A[0:3, 6:9] = R.T
        ######### Fill in the A matrix #########

        ######### Fill in the B matrix #########
        # Linear Acceleration Dynamics
        B[9:12, :] = np.hstack([np.eye(3), np.eye(3), np.eye(3), np.eye(3)]) / self.mass

        # Angular Acceleration Dynamics
        # Calculate moment of inertia in world frame
        I_moment_world_inv = R @ self.I_moment_inv @ R.T
        
        # TODO: Use horizon_idx to get the positions here
        # diff_com = self.x_curr[3:6] - trunk_trajectory[0]["p_com"]
        # com = trunk_trajectory[horizon_idx]["p_com"] + diff_com
        com = self.x_curr[3:6]
        for i in range(4):
            r = self.foot_positions[i] - com
            r_cross = np.array([
                [0, -r[2], r[1]],
                [r[2], 0, -r[0]],
                [-r[1], r[0], 0]
            ])
            B[6:9, i*3:i*3+3] = I_moment_world_inv @ r_cross

        ######### Fill in the B matrix #########
        return A, B


    def DiscreteTimeLinearizedDynamics(self, trunk_trajectory, horizon_idx):
        """
        Calculate the discrete time linearized dynamics at horizon_idx of the trajectory
        """
        Ac, Bc = self.ContinuosTimeLinearizedDynamics(trunk_trajectory, horizon_idx)
        dt = self.dt

        # Euler integration
        A = np.eye(Ac.shape[0]) + dt * Ac
        B = dt * Bc

        return A, B


        
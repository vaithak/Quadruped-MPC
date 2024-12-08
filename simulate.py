import pydot
import numpy as np
from IPython.display import SVG, display

from pydrake.all import Simulator, DiagramBuilder, AddMultibodyPlantSceneGraph,\
                        Parser, RigidTransform, MeshcatVisualizer, MeshcatVisualizerParams,\
                        HalfSpace, CoulombFriction, StartMeshcat, Box, Sphere, GeometryInstance,\
                        GeometryFrame, MakePhongIllustrationProperties, ContactVisualizer, DiscreteContactApproximation
import planner.planner as planner
import controller
# from controllers.inverse_dynamics_controller import IDController

def setup_plant_and_builder(
    urdf_path,
    ground_urdf_path,
    planner_class,
    controller_class,
    planner_mode = 2,
    dt = 8e-3,
    mpc_horizon_length = 16,
    gravity_value = 9.81,
    mu = 1.0,
    foot_clearance = 0.0175
):
    """
    Load and visualize a URDF file using Drake's MeshcatVisualizer
    
    Args:
        urdf_path (str): Path to the URDF file
        planner_class (class): Class of the trunk-model planner to use
        controller_class (class): Class of the controller to use
        dt (float): Time step for the simulation
    """
    # Start the Meshcat server
    meshcat = StartMeshcat()
    
    # Build the block diagram for the simulation
    builder = DiagramBuilder()

    # Add the multibody plant and scene graph
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)
    parser = Parser(plant)
    parser.AddModels(urdf_path)

    # Add collision geometry using ground.urdf
    parser.AddModels(ground_urdf_path)

    # Turn off gravity
    g = plant.mutable_gravity_field()
    g.set_gravity_vector([0,0,-gravity_value])

    # Finalize the plant
    plant.Finalize()

    # Add custom visualizations for the trunk frame
    frame_ids = {}
    if planner_class is not None:
        trunk_source = scene_graph.RegisterSource("trunk")
        trunk_frame = GeometryFrame("trunk")
        scene_graph.RegisterFrame(trunk_source, trunk_frame)

        # Dictionary to store frame ids
        frame_ids = {}

        # Create geometry instances
        trunk_shape = Box(0.4,0.2,0.1)
        trunk_color = np.array([0.1,0.1,0.1,0.4])
        X_trunk = RigidTransform()
        X_trunk.set_translation(np.array([0.0,0.0,0.0]))
        trunk_geometry = GeometryInstance(X_trunk,trunk_shape,"trunk")
        trunk_geometry.set_illustration_properties(MakePhongIllustrationProperties(trunk_color))
        scene_graph.RegisterGeometry(trunk_source, trunk_frame.id(), trunk_geometry)

        # Register the trunk frame
        frame_ids["trunk"] = trunk_frame.id()

        # Register the foot frames and geometry
        for foot in ["lf","rf","lh","rh"]:
            foot_frame = GeometryFrame(foot)
            scene_graph.RegisterFrame(trunk_source, foot_frame)
            foot_shape = Sphere(0.02)
            X_foot = RigidTransform()
            foot_color = np.array([0.1,0.1,0.1,0.4])
            foot_geometry = GeometryInstance(X_foot,foot_shape,foot)
            foot_geometry.set_illustration_properties(MakePhongIllustrationProperties(foot_color))
            scene_graph.RegisterGeometry(trunk_source, foot_frame.id(), foot_geometry)
            frame_ids[foot] = foot_frame.id()

    # Create high-level trunk-model planner
    planner = None
    if planner_class is not None:
        planner = builder.AddSystem(
                        planner_class(
                            frame_ids, 
                            mode=planner_mode, 
                            horizon_length=mpc_horizon_length,
                            dt=dt,
                            foot_clearance=foot_clearance
                        )
                    )

    # Add the controller
    controller = None
    if controller_class is not None:
        controller = builder.AddSystem(
            controller_class(
                plant,
                dt,
                mpc_horizon_length = mpc_horizon_length,
                gravity_value = gravity_value,
                mu = mu
            )
            # controller_class(plant, dt)
        )

    # Connect the trunk-model planner to the scene graph
    if planner is not None:
        builder.Connect(
                planner.get_output_port_by_name("trunk_geometry"),
                scene_graph.get_source_pose_port(trunk_source))

    # Connect the planner to the controller
    if controller is not None and planner is not None:
        builder.Connect(planner.get_output_port_by_name("trunk_trajectory"), 
                        controller.get_input_port_by_name("trunk_input"))

    # Connect the controller to the plant
    if controller is not None:
        builder.Connect(controller.get_output_port_by_name("quadruped_torques"),
                        plant.get_actuation_input_port())
        builder.Connect(plant.get_state_output_port(),
                        controller.get_input_port_by_name("quadruped_state"))
        # Send contact points to the controller
        builder.Connect(plant.get_contact_results_output_port(),
                        controller.get_input_port_by_name("contact_results"))

    # Add the visualizer
    vis_params = MeshcatVisualizerParams(publish_period=0.01)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params=vis_params)
    # Visualize the contact forces
    ContactVisualizer.AddToBuilder(builder, plant, meshcat)

    # Compile and plot the diagram
    diagram = builder.Build()
    display(SVG(pydot.graph_from_dot_data(
        diagram.GetGraphvizString(max_depth=2))[0].create_svg()))

    return plant, diagram, scene_graph

def simulate(plant, diagram, init_state, init_state_dot, sim_time):
    """
    Run the simulation
    
    Args:
        plant (MultibodyPlant): MultibodyPlant object
        diagram (Diagram): Block diagram for the simulation
        init_state (np.array): Initial state
        init_state_dot (np.array): Initial velocity
        sim_time (float): Simulation time
    """
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)

    # Set the robot state
    context = simulator.get_mutable_context()
    plant_context = diagram.GetMutableSubsystemContext(
            plant, context)
    print("init_state", init_state)
    print("num_positions", plant.num_positions())
    plant.SetPositions(plant_context, init_state)
    plant.SetVelocities(plant_context, init_state_dot)

    # Get current state of the plant and print it
    state = plant.GetPositionsAndVelocities(plant_context)
    print("Current state of the plant:", state)

    # Print initial coordinates of each foot
    foot_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
    for foot_name in foot_names:
        foot = plant.GetBodyByName(foot_name)
        x = foot.EvalPoseInWorld(plant_context).translation()
        print(f"Initial position of the {foot_name}:", x)

    # Simulate the robot
    simulator.AdvanceTo(sim_time)

    # Print final coordinates of each foot
    foot_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
    for foot_name in foot_names:
        foot = plant.GetBodyByName(foot_name)
        x = foot.EvalPoseInWorld(plant_context).translation()
        print(f"Final position of the {foot_name}:", x)


if __name__ == "__main__":
    # Replace with your URDF path
    urdf_path = "models/mini_cheetah.urdf"
    ground_urdf_path = "models/ground.urdf"
    
    planner_class = planner.Planner
    controller_class = controller.Controller
    # controller_class = IDController

    plant, diagram, scene_graph = setup_plant_and_builder(urdf_path, ground_urdf_path, planner_class, controller_class)
    q = np.zeros((plant.num_positions(),))
    q = np.asarray([1.0, 0.0, 0.0, 0.0,     # base orientation
                    0.0, 0.0, 0.3,          # base position
                    0.0, -0.8, 1.6,          # lf leg
                    0.0, -0.8, 1.6,          # rf leg
                    0.0, -0.8, 1.6,          # lh leg
                    0.0, -0.8, 1.6])         # rh leg
    qd = np.zeros((plant.num_velocities(),))
    
    # Initial com z-velocity - for experiments
    qd[5] = 0.0

    simulate(plant, diagram, q, qd, 20.0)


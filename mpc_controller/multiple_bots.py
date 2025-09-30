from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from absl import app
from absl import flags
import scipy.interpolate
import numpy as np
import pybullet_data as pd
from pybullet_utils import bullet_client

import time
import pybullet
import random

from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller

#uncomment the robot of choice
from mpc_controller import laikago_sim as robot_sim
#from mpc_controller import a1_sim as robot_sim

FLAGS = flags.FLAGS


from mpc_controller.quadruped_ctrl import Quadruped

from .bots.AC_unicycle import Unicycle
from .controllers.QP_controller_unicycle import QP_Controller_Unicycle

# Creating Bots

bot1_config_file_path = 'mpc_controller//bots//bot_config//bot1.json'#bot1_config_file_path = 'bots//bot_config//bot1.json'
bot2_config_file_path = 'mpc_controller//bots//bot_config//bot2.json'#bot2_config_file_path = 'bots//bot_config//bot2.json'

bot1 = Unicycle.from_JSON(bot1_config_file_path)
bot2 = Unicycle.from_JSON(bot2_config_file_path)

bot = bot1
bot_2 = bot2
SIM_FREQ = 30

CTRL0 = Quadruped(1/SIM_FREQ)

_NUM_SIMULATION_ITERATION_STEPS = 300


_STANCE_DURATION_SECONDS = [
    0.3
] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).


# Trotting
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]
_MAX_TIME_SECONDS = 100

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)


def _setup_multiple_robots(num_bots, start_positions):
    """Setup multiple robots and their controllers."""
    robots = []
    controllers = []
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # Load plane
    p.loadURDF("plane.urdf")
    
    # Setup robots and controllers
    for i in range(num_bots):
        robot_uid = p.loadURDF(robot_sim.URDF_NAME, start_positions[i])
        robot = robot_sim.SimpleRobot(p, robot_uid, simulation_time_step=0.001)
        controller = _setup_controller(robot)
        controller.reset()
        robots.append(robot)
        controllers.append(controller)
    
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    return p, robots, controllers

def _run_multiple_bots_example(num_bots, max_time=_MAX_TIME_SECONDS):
    """Run example with multiple bots."""
    
    start_positions = [[i * 1.5, 0., 0.] for i in range(num_bots)]  # Adjust spacing
    p, robots, controllers = _setup_multiple_robots(num_bots, start_positions)
    current_times = [robot.GetTimeSinceReset() for robot in robots]

    while max(current_times) < max_time:
        for i in range(num_bots):
            # Get state of the robot
            pos, orn = p.getBasePositionAndOrientation(robots[i].quadruped)
            vel, ang_v = p.getBaseVelocity(robots[i].quadruped)
            
            # Generate linear and angular speeds
            # lin_speed, ang_speed = _generate_example_linear_angular_speed(current_times[i])
            _update_controller_params(controllers[i], np.array([0.01, 0, 0]), 0)
            
            # Update controller and step simulation
            controllers[i].update()
            hybrid_action, _ = controllers[i].get_action()
            robots[i].Step(hybrid_action)
        
        # Advance time for all robots
        current_times = [robot.GetTimeSinceReset() for robot in robots]

        # Simulate a time step
        p.stepSimulation()
        time.sleep(1 / SIM_FREQ)

def _generate_example_linear_angular_speed(t):
  """Creates an example speed profile based on time for demo purpose."""
  vx = 0.2 * robot_sim.MPC_VELOCITY_MULTIPLIER
  vy = 0.2 * robot_sim.MPC_VELOCITY_MULTIPLIER
  wz = 0.8 * robot_sim.MPC_VELOCITY_MULTIPLIER
  
  time_points = (0, 2, 4, 6)
  speed_points = ((vx, 0, 0, 0), (vx, 0, 0, 0), (vx, 0, 0, 0), (vx, 0, 0, 0))

  speed = scipy.interpolate.interp1d(
      time_points,
      speed_points,
      kind="previous",
      fill_value="extrapolate",
      axis=0)(
          t)

  return speed[0:3], speed[3]


def _setup_controller(robot):
  desired_speed = (0, 0)
  desired_twisting_speed = 0

  gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
      robot,
      stance_duration=_STANCE_DURATION_SECONDS,
      duty_factor=_DUTY_FACTOR,
      initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
      initial_leg_state=_INIT_LEG_STATE)
  state_estimator = com_velocity_estimator.COMVelocityEstimator(robot,
                                                                window_size=20)
  sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_height=robot_sim.MPC_BODY_HEIGHT,
      foot_clearance=0.01)

  st_controller = torque_stance_leg_controller.TorqueStanceLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_body_height=robot_sim.MPC_BODY_HEIGHT,
      body_mass=robot_sim.MPC_BODY_MASS,
      body_inertia=robot_sim.MPC_BODY_INERTIA)

  controller = locomotion_controller.LocomotionController(
      robot=robot,
      gait_generator=gait_generator,
      state_estimator=state_estimator,
      swing_leg_controller=sw_controller,
      stance_leg_controller=st_controller,
      clock=robot.GetTimeSinceReset)
  return controller



def _update_controller_params(controller, lin_speed, ang_speed):
  controller.swing_leg_controller.desired_speed = lin_speed
  controller.swing_leg_controller.desired_twisting_speed = ang_speed
  controller.stance_leg_controller.desired_speed = lin_speed
  controller.stance_leg_controller.desired_twisting_speed = ang_speed




def main(argv):
    del argv
    _run_multiple_bots_example(num_bots=3) 


if __name__ == "__main__":
  app.run(main)
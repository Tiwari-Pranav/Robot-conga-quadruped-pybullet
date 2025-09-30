import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev, splder

import pybullet
from absl import app
import pybullet_data as pd
from pybullet_utils import bullet_client

from mpc_controller import locomotion_controller
from mpc_controller import com_velocity_estimator
from mpc_controller import openloop_gait_generator
from mpc_controller import laikago_sim as robot_sim
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller
from mpc_controller import gait_generator as gait_generator_lib



class bot_sim(bullet_client.BulletClient):
    
    def __init__(self, num_bots, dt):
        super().__init__(connection_mode=pybullet.GUI)

         # Parameters for control and trajectory generation
        self.y = 1.5 # Gamma parameter
        self.a = 5.0 # a parameter
        self.b = 1.0 # b parameter
        self.v = 0.5 # linear velocitiy

        self.dt = dt # time step
        self.gx = lambda x : 0. # tajectory y=g(x)
        self.d_gx = lambda x : 0. # first derivative of trajector w.r.t x
        self.d2_gx = lambda x : 0. # second derivative of trajectory w.r.t x

        self.goal = np.zeros(2) # Target position for robots directed by arrow keys upto which we need to interpolate the trajectory
        self.num_bots = num_bots # number of bots
        self.curr_vel = np.zeros((num_bots, 2)) # current velocity of bots [linear, angular]

        # current positions of bots [x, y, theta].T
        self.curr_pose = np.zeros((self.num_bots, 3)) 
        self.curr_pose[:, 0] += 1.5*np.arange(self.num_bots-1, -1, -1)
        self.curr_pose[:, 1] = self.gx(self.curr_pose[:, 0])
        self.curr_pose[:, 2] = np.arctan(self.d_gx(self.curr_pose[:, 0]))

        # desired positions of bots [x*, y*, theta*].T
        self.desired_pose = np.zeros_like(self.curr_pose)
        self.desired_pose[:, 0] = self.curr_pose[:, 0] + 1e-1
        self.desired_pose[:, 1] = self.gx(self.desired_pose[:, 0])
        self.desired_pose[:, 2] = np.arctan(self.d_gx(self.desired_pose[:, 0]))

        # position error for bots 
        self.error = np.zeros_like(self.curr_pose)

        
    
    def update_traj(self, sign):
        """
        Updates the trajectory functions of the robots based on the sign i.e. the direction you want to turn.

        Parameters:
        sign (int): The sign indicating the direction of the trajectory update.
            A positive sign indicates a rightward trajectory update, while a negative sign indicates a leftward trajectory update.

        Returns:
        None. The function updates the trajectory functions (gx, d_gx, d2_gx).
        """
        goal_angle = self.curr_pose[0, 2] + sign*4e-1 # at what angle do we need the next point
        if np.abs(goal_angle) > 0.43*np.pi: return  

        # goal position:: [desired_pose_x+2.5*cos(desired angle) desired_pose_y+2.5*sin(desired angle)]
        self.goal = (2.5*np.array([np.cos(goal_angle), 
                                  np.sin(goal_angle)])
                     + self.desired_pose[0, :2]) 

        x_coord = np.concatenate((self.desired_pose[::-1, 0], [self.goal[0]]))
        y_coord = np.concatenate((self.desired_pose[::-1, 1], [self.goal[1]]))

        # Interpolate the trajectory upto the goal position
        spline_tup = splrep(x_coord, y_coord, s=0)
        spline_tup_der = splder(spline_tup, n=1)
        spline_tup_der2 = splder(spline_tup, n=2)

        self.gx = lambda x : splev(x, spline_tup)
        self.d_gx = lambda x : splev(x, spline_tup_der)
        self.d2_gx = lambda x : splev(x, spline_tup_der2)

        
    
    def get_curr_err(self):
        """
        Updates the current position and error of the robots.

        This function calculates the current position of each robot in the simulation,
        updates the desired position based on the trajectory functions, and then calculates
        the error between the current and desired positions.

        Parameters:
        None. The function uses the current state of the robots, trajectory functions, and
        simulation parameters stored in the class instance.

        Returns:
        None. The function updates the 'curr_pose', 'desired_pose', and 'error' attributes
        of the class instance.
        """
        for i in range(self.num_bots):
            pos_orien = self.getBasePositionAndOrientation(self.bots[i].quadruped) # get the current position of ith quaroped robot in the simulation [[x,y,z],[orientation]] 
            # position format:: [x ,y , theta]
            self.curr_pose[i,:] = np.array([pos_orien[0][0], pos_orien[0][1],
                                            self.getEulerFromQuaternion(pos_orien[1])[-1]]) # Using getEulerFromQuaternion to convert the quaternion to Euler 

        # Update the desired pose and orientation for visual bots
        # Distance travelled in time step dt along the time parameterized path :: v_linear * dt
        # cos(theta) = 1/[1 + tan^2(theta)] i.e.tan(theta) = dg/gx 
        self.desired_pose[:, 0] += self.v*self.dt/np.sqrt(1 + np.square(self.d_gx(self.desired_pose[:, 0]))) # x = [v_linear * dt] * cos (theta)
        self.desired_pose[:, 1] = self.gx(self.desired_pose[:, 0]) # y=g(x)
        self.desired_pose[:, 2] = np.arctan(self.d_gx(self.desired_pose[:, 0])) # theta = arctan (dg/dx)
        # Update error variables 
        self.error = np.copy(self.curr_pose - self.desired_pose)

    
    
    def get_ctrl_input(self):
        """
        Calculates the control inputs (linear and angular velocities) for each robot based on the current error and trajectory.

        Parameters: None. the function only uses the instance of the bot_sim class, and its arributes such as the linear velocity, the gamma parameter, 
        the current position error of the robots, the desired position of the robots, the current position of the robots, 
        the first derivative of the trajectory function g(x), the second derivative of the trajectory function g(x), the a parameter,
        the b parameter.

        Returns:
        None. The function updates the 'curr_vel' attribute of the class instance with the calculated control inputs.
        """
        # u1* = v_linear 
        # u2* refer to equation 8 of paper
        self.curr_vel[:, 0] = ((self.v*np.cos(self.desired_pose[:, -1]))
                                -self.y*self.error[:, 0])/np.cos(self.curr_pose[:, 2]) # Please refer to equation (6) of paper

        self.curr_vel[:, 1] = (self.curr_vel[:, 0]*self.d2_gx(self.curr_pose[:, 0])
                               /np.power(1 + np.square(self.d_gx(self.curr_pose[:, 0])), 1.5)
                               - self.a*self.error[:, 1] - self.b*self.error[:, 2]) # Please refer to equation (6) of paper

        
        
    def setup_controller(self, index):
        """
        Sets up and initializes a locomotion controller for a specific robot.

        Parameters:
        index (int): The index of the robot for which the controller needs to be set up.

        Returns:
        controller (LocomotionController): The initialized controller for the specified robot.
        """
        DUTY_FACTOR = [0.6] * 4
        STANCE_DURATION_SECONDS = [0.3] * 4
        INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]
        INIT_LEG_STATE = (gait_generator_lib.LegState.SWING,
                          gait_generator_lib.LegState.STANCE,
                          gait_generator_lib.LegState.STANCE,
                          gait_generator_lib.LegState.SWING)

        desired_speed = np.zeros(2)
        desired_twisting_speed = 0.

        gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
            self.bots[index],
            stance_duration=STANCE_DURATION_SECONDS,
            duty_factor=DUTY_FACTOR,
            initial_leg_phase=INIT_PHASE_FULL_CYCLE,
            initial_leg_state=INIT_LEG_STATE)
        state_estimator = com_velocity_estimator.COMVelocityEstimator(self.bots[index], window_size=20)

        sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
            self.bots[index],
            gait_generator,
            state_estimator,
            desired_speed=desired_speed,
            desired_twisting_speed=desired_twisting_speed,
            desired_height=robot_sim.MPC_BODY_HEIGHT,
            foot_clearance=0.01)

        st_controller = torque_stance_leg_controller.TorqueStanceLegController(
            self.bots[index],
            gait_generator,
            state_estimator,
            desired_speed=desired_speed,
            desired_twisting_speed=desired_twisting_speed,
            desired_body_height=robot_sim.MPC_BODY_HEIGHT,
            body_mass=robot_sim.MPC_BODY_MASS,
            body_inertia=robot_sim.MPC_BODY_INERTIA)

        controller = locomotion_controller.LocomotionController(
            robot=self.bots[index],
            gait_generator=gait_generator,
            state_estimator=state_estimator,
            swing_leg_controller=sw_controller,
            stance_leg_controller=st_controller,
            clock=self.bots[index].GetTimeSinceReset)

        return controller

    

    def setup_multiple_robots(self):
        """
        Sets up and initializes multiple robots in the simulation.

        This function loads a plane URDF, sets up the physics engine parameters,
        sets up the gravity, time step, and real-time simulation. It then loads
        multiple robot URDFs, creates SimpleRobot instances for each robot,
        sets up and initializes a locomotion controller for each robot, and
        retrieves the current times.

        Parameters:
        None. The function uses the instance attributes of the bot_sim class,
        such as self.setAdditionalSearchPath, self.configureDebugVisualizer,
        self.setPhysicsEngineParameter, self.setRealTimeSimulation,
        self.setGravity, self.setTimeStep, self.loadURDF, self.bots,
        self.controllers, self.curr_pose, self.num_bots, self.dt,
        robot_sim.URDF_NAME, self.getQuaternionFromEuler,
        self.get_current_times, and self.configureDebugVisualizer.

        Returns:
        None. The function sets up and initializes multiple robots in the simulation.
        """
        self.setAdditionalSearchPath(pd.getDataPath())
        self.configureDebugVisualizer(self.COV_ENABLE_GUI, 0)
        self.configureDebugVisualizer(self.COV_ENABLE_RENDERING, 0)
        self.setPhysicsEngineParameter(enableConeFriction=0)
        # self.setPhysicsEngineParameter(numSolverIterations=30)

        self.setRealTimeSimulation(0)
        self.setGravity(0, 0, -9.81)
        self.setTimeStep(self.dt)

        # Load plane
        self.loadURDF("plane.urdf")

        # Setup robots and controllers
        self.bots = [] # list of bot objects
        self.controllers = [] # list of controller objects

        for i in range(self.num_bots):
            # TheloadURDF sends a command to the physics server to load bot physics model from the URDF.
            robot_uid = self.loadURDF(robot_sim.URDF_NAME, [self.curr_pose[i, 0], self.curr_pose[i, 1], 0.4],
                                      self.getQuaternionFromEuler([0., 0., self.curr_pose[i, 2]])) # each bot is assigned with a unique non-negative body id and if load fails it returns a negative value 
            self.bots.append(robot_sim.SimpleRobot(self, robot_uid, simulation_time_step=self.dt)) 

            controller = self.setup_controller(i) # setup the controller for speified bot
            controller.reset() # reset clock of the bot and its attributes
            self.controllers.append(controller) # add the controller object to list

        self.get_current_times()
        self.configureDebugVisualizer(self.COV_ENABLE_RENDERING, 1)




    def control_bots(self):
        """
        This function controls the movement of each robot in the simulation.
        It iterates over each robot, sets the desired speed and twisting speed for the swing and stance leg controllers,
        updates the controller, retrieves the hybrid action, and steps the robot with the hybrid action.

        Parameters:
        None. The function uses the attributes of the bot_sim class, such as num_bots, curr_vel,
        controllers, and bots.

        Returns:
        None. The function controls the movement of the robots in the simulation.
        """
        for i in range(self.num_bots):
            lin_speed = np.array([self.curr_vel[i, 0], 0., 0.]) # setup linear velocity
            ang_speed = self.curr_vel[i, 1] # setup angular velocity

            self.controllers[i].swing_leg_controller.desired_speed = lin_speed
            self.controllers[i].swing_leg_controller.desired_twisting_speed = ang_speed
            self.controllers[i].stance_leg_controller.desired_speed = lin_speed
            self.controllers[i].stance_leg_controller.desired_twisting_speed = ang_speed

            self.controllers[i].update()
            hybrid_action, _ = self.controllers[i].get_action()
            self.bots[i].Step(hybrid_action)

        
     
    def get_current_times(self):
        """
        Retrieves the current times for all robots in the simulation.

        Parameters:
        None. The function uses the 'bots' attribute of the 'bot_sim' class, which is a list of robot objects.

        Returns:
        None. The function updates the 'current_times' attribute of the 'bot_sim' class with an array of current times for all robots.
        The current time for each robot is obtained using the 'GetTimeSinceReset' method of the robot object.
        """
        self.current_times = np.array([bot.GetTimeSinceReset() for bot in self.bots])
        
        
    def run_multiple_bots_example(self):
        """
        This function is the main entry point for running the simulation with multiple robots.
        It sets up the robots, initializes the controllers, and then enters a loop to control the robots.
        The function also handles keyboard events to update the trajectory of the robots.
        In case of any exceptions, it stops the simulation and displays the error message.
        """
        self.setup_multiple_robots()

        try:
            err = []
            cmd_vel = []
            cmd_keys =  [self.B3G_LEFT_ARROW, self.B3G_RIGHT_ARROW, self.B3G_UP_ARROW] # The control keys to change trajectory of bots

            running = True
            while running:
                self.get_curr_err()
                self.get_ctrl_input()
                self.control_bots()

                self.resetDebugVisualizerCamera(5, -90, -89,
                                                [self.curr_pose[self.num_bots//2, 0],
                                                 self.curr_pose[self.num_bots//2, 1],
                                                 0.])

                keys = self.getKeyboardEvents() # receive all keyboard events since the last time you called
                # Update the trjectory by passing passing sign based on arrow keys
                if cmd_keys[0] in keys and keys[cmd_keys[0]] & self.KEY_WAS_TRIGGERED: self.update_traj(1) 
                if cmd_keys[1] in keys and keys[cmd_keys[1]] & self.KEY_WAS_TRIGGERED: self.update_traj(-1)
                if cmd_keys[2] in keys and keys[cmd_keys[2]] & self.KEY_WAS_TRIGGERED: self.update_traj(0)

                self.stepSimulation()
                self.get_current_times()
                time.sleep(self.dt)

                err.append(np.copy(self.error))
                cmd_vel.append(np.copy(self.curr_vel))

        except:
            running = False
            print("Exiting...")

            if self.isConnected():  self.disconnect()
            _, ax = plt.subplots(2, 2, figsize=(10, 7))
            
            # Graph plotting code
            err = np.array(err)
            cmd_vel = np.array(cmd_vel)

            for i in range(self.num_bots):
                ax[0][0].plot(np.linalg.norm(err[:, i, :2], axis=1))
                ax[0][1].plot(err[:, i, 2])
                ax[1][0].plot(cmd_vel[:, i, 0])
                ax[1][1].plot(cmd_vel[:, i, 1])

            ax[0][0].set_ylabel('Positional error (m)')
            ax[0][1].set_ylabel('Angular error (rad)')
            ax[1][0].set_ylabel('Linear Command Velocity (m/s)')
            ax[1][1].set_ylabel('Angular Command Velocity (rad/s)')

            plt.show()


def main(argv):
    del argv
    B = bot_sim(5, 7e-3)
    B.run_multiple_bots_example()
    

if __name__ == '__main__':
    app.run(main)

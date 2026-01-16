"""Example utility module.

Please use a file like this one to add extra functions.

"""

def exampleFunction():
    """Example of user-defined function.

    """
    x = -1
    return x

def planning_example(self, use_firmware, initial_info):
    """Trajectory planning algorithm"""
    #########################
    # REPLACE THIS (START) ##
    #########################
    ## generate waypoints for planning

    # Call a function in module `example_custom_utils`.
    ecu.exampleFunction()

    # initial waypoint
    if use_firmware:
        waypoints = [(self.initial_obs[0], self.initial_obs[2], initial_info["gate_dimensions"]["tall"]["height"])]  # Height is hardcoded scenario knowledge.
    else:
        waypoints = [(self.initial_obs[0], self.initial_obs[2], self.initial_obs[4])]

    # Example code: hardcode waypoints 
    waypoints.append((-0.5, -3.0, 2.0))
    waypoints.append((-0.5, -2.0, 2.0))
    waypoints.append((-0.5, -1.0, 2.0))
    waypoints.append((-0.5,  0.0, 2.0))
    waypoints.append((-0.5,  1.0, 2.0))
    waypoints.append((-0.5,  2.0, 2.0))
    waypoints.append([initial_info["x_reference"][0], initial_info["x_reference"][2], initial_info["x_reference"][4]])

    # Polynomial fit.
    self.waypoints = np.array(waypoints)
    deg = 6
    t = np.arange(self.waypoints.shape[0])
    fx = np.poly1d(np.polyfit(t, self.waypoints[:,0], deg))
    fy = np.poly1d(np.polyfit(t, self.waypoints[:,1], deg))
    fz = np.poly1d(np.polyfit(t, self.waypoints[:,2], deg))
    duration = 15
    t_scaled = np.linspace(t[0], t[-1], int(duration*self.CTRL_FREQ))
    self.ref_x = fx(t_scaled)
    self.ref_y = fy(t_scaled)
    self.ref_z = fz(t_scaled)

    #########################
    # REPLACE THIS (END) ####
    #########################

    return t_scaled

def cmdFirmware_example(self,
                time,
                obs,
                reward=None,
                done=None,
                info=None
                ):
    """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

    INSTRUCTIONS:
        Re-implement this method to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
        from Crazyswarm to the Crazyflie using, e.g., a `cmdFullState` call.

    Args:
        time (float): Episode's elapsed time, in seconds.
        obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
        reward (float, optional): The reward signal.
        done (bool, optional): Wether the episode has terminated.
        info (dict, optional): Current step information as a dictionary with keys
            'constraint_violation', 'current_target_gate_pos', etc.

    Returns:
        Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
        List: arguments for the type of command (see comments in class `Command`)

    """
    if self.ctrl is not None:
        raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")

    # [INSTRUCTIONS] 
    # self.CTRL_FREQ is 30 (set in the getting_started.yaml file) 
    # control input iteration indicates the number of control inputs sent to the quadrotor
    iteration = int(time*self.CTRL_FREQ)

    #########################
    # REPLACE THIS (START) ##
    #########################

    # print("The info. of the gates ")
    # print(self.NOMINAL_GATES)

    if iteration == 0:
        height = 1
        duration = 2

        command_type = Command(2)  # Take-off.
        args = [height, duration]

    # [INSTRUCTIONS] Example code for using cmdFullState interface   
    elif iteration >= 3*self.CTRL_FREQ and iteration < 20*self.CTRL_FREQ:
        step = min(iteration-3*self.CTRL_FREQ, len(self.ref_x) -1)
        target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
        target_vel = np.zeros(3)
        target_acc = np.zeros(3)
        target_yaw = 0.
        target_rpy_rates = np.zeros(3)

        command_type = Command(1)  # cmdFullState.
        args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

    elif iteration == 20*self.CTRL_FREQ:
        command_type = Command(6)  # Notify setpoint stop.
        args = []

    # [INSTRUCTIONS] Example code for using goTo interface 
    elif iteration == 20*self.CTRL_FREQ+1:
        x = self.ref_x[-1]
        y = self.ref_y[-1]
        z = 1.5 
        yaw = 0.
        duration = 2.5

        command_type = Command(5)  # goTo.
        args = [[x, y, z], yaw, duration, False]

    elif iteration == 23*self.CTRL_FREQ:
        x = self.initial_obs[0]
        y = self.initial_obs[2]
        z = 1.5
        yaw = 0.
        duration = 6

        command_type = Command(5)  # goTo.
        args = [[x, y, z], yaw, duration, False]

    elif iteration == 30*self.CTRL_FREQ:
        height = 0.
        duration = 3

        command_type = Command(3)  # Land.
        args = [height, duration]

    elif iteration == 33*self.CTRL_FREQ-1:
        command_type = Command(4)  # STOP command to be sent once the trajectory is completed.
        args = []

    else:
        command_type = Command(0)  # None.
        args = []

    #########################
    # REPLACE THIS (END) ####
    #########################

    return command_type, args
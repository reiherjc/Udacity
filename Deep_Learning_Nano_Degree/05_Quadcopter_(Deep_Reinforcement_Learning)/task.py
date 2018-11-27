import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, target_v=None, target_angular_v=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0,0,1,0,0,0]) #mod: z-coord set from 10 to 1
        self.target_v= target_v
        self.target_angular_v = target_angular_v
        
    def get_reward(self,done):
        """Uses current pose of sim to return reward."""
        
        ###########################################        
        # My reward function from 1st submission: #
        ###########################################
        #reward = 1.-(((self.sim.pose - self.target_pos)**2).sum())**.5 - ((((self.sim.v-self.target_v)**2).sum())**.5)- ((((self.sim.angular_v-self.target_angular_v)**2).sum())**.5)
        #return reward
        
        #########################################################################
        # Reward function from 2nd submission                                   #
        #########################################################################
        #dist_from_target =  (((self.sim.pose - self.target_pos)**2).sum())**.5 #- ((((self.sim.v-self.target_v)**2).sum())**.5
        #crash = False 
        #if self.sim.pose[2]<=0:
            #crash=True
                                                                                                                                                         
        #if dist_from_target < 2:  #if copter is close to target position
            #return 0.5, dist_from_target
        #elif crash: # if copter crashes i.e. hits the line z=0
            #return -1., dist_from_target
        #else: #if copter is far waway from target position
            #return -0.5, dist_from_target
        
        
        ###############################################################################
        #Reward function for 3rd (current) submission                                 #
        ###############################################################################
                
        # Reward is given for coming closer to desired height
        reward = -abs(self.sim.pose[2] - self.target_pos[2])
        
        #extra reward is given for staying within 3m distance from target altitude
        if self.sim.pose[2] < self.target_pos[2] + 3 and self.sim.pose[2] > self.target_pos[2] - 3:
            reward+=10
        # extra reward is given if the agent stays exactly at the target altitude
        if(self.sim.pose[2] == self.target_pos[2]):
            reward += 20.0
        
        # Crashing is penalized
        if done  and self.sim.time < self.sim.runtime: 
            reward = -30
        return reward
    
                                                                            
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward = self.get_reward(done)
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
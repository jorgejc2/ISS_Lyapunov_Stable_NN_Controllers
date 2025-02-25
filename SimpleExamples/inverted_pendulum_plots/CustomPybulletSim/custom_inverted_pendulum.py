import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces
import time

class RealTimeFixedInvertedPendulumEnv(gym.Env):
    """
    A real-time fixed-base inverted pendulum environment.
    Observations: [angle, angular_velocity]
    Actions: scalar torque about the y-axis
    """

    def __init__(self):
        super(RealTimeFixedInvertedPendulumEnv, self).__init__()

        # Connect to PyBullet in GUI mode.
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Gravity
        p.setGravity(0, 0, -9.8)
        # Enable real-time simulation
        p.setRealTimeSimulation(1)

        # Pendulum parameters
        self.pendulum_length = 1.0
        self.pendulum_radius = 0.05

        # Create collision & visual shapes (capsule)
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_CAPSULE,
            radius=self.pendulum_radius,
            height=self.pendulum_length
        )
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_CAPSULE,
            radius=self.pendulum_radius,
            length=self.pendulum_length,
            rgbaColor=[0.8, 0.2, 0.2, 1]
        )

        # Position so bottom (pivot) is at z=0
        base_position = [0, 0, self.pendulum_length / 2]
        self.pendulum_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=base_position
        )

        # Fix the bottom of the pendulum
        pivot_in_body = [0, 0, -self.pendulum_length / 2]
        pivot_in_world = [0, 0, 0]
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.pendulum_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=pivot_in_body,
            childFramePosition=pivot_in_world
        )

        # Action/observation spaces
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # Optional state saving
        self.stateId = -1

    def reset(self):
        if not p.isConnected():
            self.physicsClient = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.8)
            p.setRealTimeSimulation(1)

        # Reset near upright
        init_angle = np.random.uniform(-0.05, 0.05)
        orientation = p.getQuaternionFromEuler([init_angle, 0, 0])
        base_position = [0, 0, self.pendulum_length / 2]
        p.resetBasePositionAndOrientation(self.pendulum_id, base_position, orientation)
        p.resetBaseVelocity(self.pendulum_id, [0, 0, 0], [0, 0, 0])

        obs = self._get_observation()
        if self.stateId < 0:
            self.stateId = p.saveState()
        return obs

    def step(self, action):
        # Apply torque about y-axis.
        torque = [0, action[0], 0]
        p.applyExternalTorque(self.pendulum_id, -1, torqueObj=torque, flags=p.LINK_FRAME)

        # Since real-time is on, we don't call p.stepSimulation() here.
        # Just sleep a tiny bit so the environment has time to visually update
        # time.sleep(1.0 / 60.0)  # ~60 FPS for real-time feel

        # Evaluate state and reward
        obs = self._get_observation()
        reward = 1.0 - abs(obs[0]) / np.pi
        done = (abs(obs[0]) > np.pi / 2)
        return obs, reward, done, {}

    def _get_observation(self):
        _, orn = p.getBasePositionAndOrientation(self.pendulum_id)
        euler = p.getEulerFromQuaternion(orn)
        angle = euler[0]
        _, angular_vel = p.getBaseVelocity(self.pendulum_id)
        angular_velocity = angular_vel[0]
        return np.array([angle, angular_velocity], dtype=np.float32)

    def render(self, mode="human"):
        # Adjust camera
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, self.pendulum_length / 2]
        )

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    # Quick demo
    env = RealTimeFixedInvertedPendulumEnv()
    obs = env.reset()
    done = False
    while not done:
        # For demonstration, apply zero torque
        obs, reward, done, info = env.step([0])
        env.render()

    env.close()

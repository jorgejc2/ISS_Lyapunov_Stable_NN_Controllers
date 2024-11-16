from typing import Union, Tuple, Optional
import scipy
import numpy as np
from numpy import ndarray
import torch
import torch.nn as nn
from torch import Tensor

# pybullet imports
import pybullet as p
import time
import pybullet_data

urdf_path = "./pybullet_urdf/"

def main():
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF(urdf_path + "plane.urdf")
    startPos = [0, 0, 1]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    # boxId = p.loadURDF(urdf_path + "r2d2.urdf", startPos, startOrientation)
    boxId = p.loadURDF(urdf_path + "quadrotor.urdf", startPos, startOrientation)
    # set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
    for i in range(10000):
        p.stepSimulation()
        time.sleep(1. / 240.)
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    print(cubePos, cubeOrn)
    p.disconnect()

if __name__ == "__main__":
    main()
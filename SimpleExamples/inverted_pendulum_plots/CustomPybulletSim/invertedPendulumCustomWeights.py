import os, inspect
import gym
import numpy as np
import pybullet_envs
import time
import torch
import torch.nn as nn

# from custom_inverted_pendulum import CustomInvertedPendulum

to_numpy = lambda x : x.detach().cpu().numpy()

def relu(x):
  return np.maximum(x, 0)

class SmallReactivePolicy:
  "Simple multi-layer perceptron policy, no internal state"

  def __init__(self, observation_space, action_space, weights_dir):
    # Load custom weights
    self.weights_dense1_w = np.load(os.path.join(weights_dir, "weights_layer_0.npy"))
    self.weights_dense1_b = np.load(os.path.join(weights_dir, "biases_layer_0.npy"))
    self.weights_dense2_w = np.load(os.path.join(weights_dir, "weights_layer_1.npy"))
    self.weights_dense2_b = np.load(os.path.join(weights_dir, "biases_layer_1.npy"))
    self.weights_final_w = np.load(os.path.join(weights_dir, "weights_layer_2.npy"))
    self.weights_final_b = np.load(os.path.join(weights_dir, "biases_layer_2.npy"))

	# Debugging shapes
    # print("observation_space.shape:", observation_space.shape)
    # print("action_space.shape:", action_space.shape)
    # print("weights_dense1_w.shape:", self.weights_dense1_w.shape)
    # print("weights_dense2_w.shape:", self.weights_dense2_w.shape)
    # print("weights_final_w.shape:", self.weights_final_w.shape)

	# Ensure the input/output dimensions match the environment
    assert self.weights_dense1_w.shape[0] == observation_space.shape[0], \
        f"Mismatch: weights_dense1_w.shape[0] = {self.weights_dense1_w.shape[0]} and observation_space.shape[0] = {observation_space.shape[0]}"
    assert self.weights_final_w.shape[1] == action_space.shape[0], \
        f"Mismatch: weights_final_w.shape[1] = {self.weights_final_w.shape[1]} and action_space.shape[0] = {action_space.shape[0]}"

    def act(self, ob):
        x = ob
        x = relu(np.dot(x, self.weights_dense1_w) + self.weights_dense1_b)
        x = relu(np.dot(x, self.weights_dense2_w) + self.weights_dense2_b)
        x = np.dot(x, self.weights_final_w) + self.weights_final_b
        return x

class SimpleNNController(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):

        super(SimpleNNController, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, x):
        return self.model(x)

def main():
  print("Create environment")
  # env = gym.make("InvertedPendulumBulletEnv-v0")

  from custom_inverted_pendulum import RealTimeFixedInvertedPendulumEnv
  env = RealTimeFixedInvertedPendulumEnv()


  env.render(mode="human")

  # Path to your custom weights directory
  weight_dir = os.path.join(os.path.dirname(__file__), "customWeightsOld")
  # pi = SmallReactivePolicy(env.observation_space, env.action_space, weight_dir)
  pi = SimpleNNController(env.observation_space.shape[0], env.action_space.shape[0])




  # print(f"Expected input size: {env.observation_space.shape[0]}")
  # print(f"Expected output size: {env.action_space.shape[0]}")

  # print("Model structure:")
  # print(pi)

  # file_path = os.path.join(weight_dir, "nn_controller.pth")
  # checkpoint = torch.load(file_path)
  # for key, value in checkpoint.items():
  #   print(f"Checkpoint layer: {key}, shape: {value.shape}")



  pi.load_state_dict(torch.load(os.path.join(weight_dir, "nn_controller.pth")))

  while 1:
    frame = 0
    score = 0
    restart_delay = 0
    obs = env.reset()
    print("frame")
    while 1:
      time.sleep(1. / 60.)
      a = pi(torch.from_numpy(obs).to(torch.float32))
      a = to_numpy(a)
      obs, r, done, _ = env.step(a)
      score += r
      frame += 1
      still_open = env.render(mode="human")
      if not done: 
        continue
      if restart_delay == 0:
        print("score=%0.2f in %i frames" % (score, frame))
        restart_delay = 60 * 2  # 2 sec at 60 fps
      else:
        restart_delay -= 1
        if restart_delay == 0: 
          break

if __name__ == "__main__":
  main()
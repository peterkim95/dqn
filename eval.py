import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
import gym

env = gym.make('CartPole-v0')
env.reset()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = T.Compose([T.ToTensor()])

def get_screen():
    rgb_array = env.render(mode='rgb_array').copy()
    if rgb_array.shape == (800, 1200, 3):
        rgb_array = np.resize(rgb_array, (400, 600, 3))
    assert rgb_array.shape == (400, 600, 3)
    return transform(rgb_array).unsqueeze(0) # add batch dim

class DQN(nn.Module):
    '''
    conv2d: (N, C, H, W) -> (N, C_out, H_out, W_out)
    '''
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

screen_height, screen_width = 400, 600

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
policy_net.load_state_dict(torch.load('cartpole_e100.pth', map_location=device))
policy_net.eval()

for e in range(20):
    env.reset()
    state = get_screen() - get_screen()
    while True:
        action = policy_net(state).max(1)[1].view(1,1)
        previous_screen = get_screen()
        obs, reward, done, info = env.step(action.item())

        env.render()

        next_state = get_screen() - previous_screen

        state = next_state

        if done:
            break
import gym
import math
import random
import numpy as np
import sys
import gc

from collections import namedtuple, deque
from tqdm import trange
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

if sys.argv[1] == 'ec2':
    from pyvirtualdisplay import Display
    dis = Display(visible=0, size=(1000, 1000))
    dis.start()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        # deque: Doubly Ended Queue
    
    def push(self, *args):
        '''Save a transition'''
        if len(self.memory) == self.capacity:
            print('max memory reached, will pop n lock')
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


env = gym.make('CartPole-v0').unwrapped
env.reset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = T.Compose([T.ToTensor(), T.Resize((400, 600))])
'''
ToTensor():
Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
'''

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)

# def get_screen():
#     rgb_array = np.ascontiguousarray(env.render(mode='rgb_array'), dtype=np.float32)
#     # if rgb_array.shape == (800, 1200, 3):
#     #     rgb_array = np.resize(rgb_array, (400, 600, 3))
#     rgb_tensor = transform(rgb_array).to(device).unsqueeze(0)
#     return rgb_tensor


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


UPDATES_PER_EPOCH = 50
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
# screen_height, screen_width = 400, 600

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

updates_done = 0


writer = SummaryWriter()
episode_rewards = []

num_episodes = 2000


def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() >= eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1,1) # Use policy net here...to decide best action so far
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    
    batch = Transition(*zip(*transitions)) # concatenated along field axes
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).to(device)
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    target_net.eval()
    with torch.no_grad():
        target_net_values = target_net(non_final_next_states).max(1)[0]
    next_state_values[non_final_mask] = target_net_values
    next_state_values = next_state_values.unsqueeze(1)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


for i_episode in trange(num_episodes):
    env.reset()

    current_screen, last_screen = get_screen(), get_screen()
    state = current_screen - last_screen # dummy state

    episode_reward = 0
    while True:
        action = select_action(state)
        
        _, reward, done, _ = env.step(action.item())
        episode_reward += reward
        reward = torch.tensor([[reward]], device=device)

        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
        
        memory.push(state, action, next_state, reward)
        
        state = next_state
        
        optimize_model()

        updates_done += 1

        if updates_done % UPDATES_PER_EPOCH == 0:
            epoch = updates_done // UPDATES_PER_EPOCH
            avg_reward = np.mean(episode_rewards)
            # global avg_reward
            # avg_reward = avg_reward + (episode_reward - avg_reward) / (epoch + 1)
            # plot_reward(epoch, avg_reward)
            writer.add_scalar('Avg. Reward/cartpole', avg_reward, epoch)

        if done:
            episode_rewards.append(episode_reward)
            break

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if i_episode % 50 == 0:
        torch.save(policy_net.state_dict(), f'cartpole_e{i_episode}.pt')

print('if you see this for reals then you made it')
env.render()
env.close()
# dis.stop()

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from knuckleEnv import knuckle
from tqdm import tqdm
import os

# Make sure that gpu can be used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parameters
num_episodes = 100000
batch_size = 128
learning_rate = 0.001
gamma = 0.99

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device is set to {device}')

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(19, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16,3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x
    

def compute_disc_rew(rewards):
    """add_rew = 0
    for i in reversed(range(len(rewards))):
        if rewards[i] == 0:
            add_rew = 0
        else:
            add_rew = add_rew * gamma + rewards[i]
            rewards[i] = add_rew
    """

    # Return ormalized rewards
    return F.normalize(rewards, dim=0)
        

def main():

    # History book
    state_board1 = torch.tensor([]).to(device)
    action_board1 = torch.tensor([]).to(device)
    reward_board1 = torch.tensor([]).to(device)

    state_board2 = torch.tensor([]).to(device)
    action_board2 = torch.tensor([]).to(device)
    reward_board2= torch.tensor([]).to(device)
    
    env = knuckle()
    policy_net = PolicyNetwork().to(device)
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)
    
    for e in tqdm(range(num_episodes)):
        
        state, reward, done = env.reset()
        

        while not done:
            ###### FIRST TURN ######
            # Prepare state and sample action
            state = np.append(np.array(state[0] + state[1]).flatten(), env.last_roll)
            state = torch.tensor([state], dtype=torch.float32, requires_grad=True).to(device)
            state_board1 = torch.cat((state_board1, state))
            # state_board1.append(state) # Save current state

            # Sample action
            probs = policy_net(state)
            m = Categorical(probs)
            action = m.sample()

            # Run step and save action and reward for the current state
            state, reward, done = env.step(action.cpu(), 1)
            
            # action_board1.append(action)
            # reward_board1.append(reward)
            action_board1 = torch.cat((action_board1, torch.tensor([action]).to(device)), 0)
            reward_board1 = torch.cat((reward_board1, torch.tensor([reward]).to(device)), 0)
            if done:
                break

            ###### SECOND TURN ######
            # Reverse board position to encode the state
            state = np.append(np.array(state[1] + state[0]).flatten(), env.last_roll)
            state = torch.tensor([state], dtype=torch.float32, requires_grad=True).to(device)
            # state_board2.append(state)
            state_board2 = torch.cat((state_board2, state), 0)

            # Sample action
            probs = policy_net(state)
            m = Categorical(probs)
            action = m.sample()

            # Run step and save action and reward for the current state
            state, reward, done = env.step(action.cpu(), 2)
            
            # action_board2.append(action)
            # reward_board2.append(reward)
            action_board2 = torch.cat((action_board2, torch.tensor([action]).to(device)), 0)
            reward_board2 = torch.cat((reward_board2, torch.tensor([reward]).to(device)), 0)

        # env.display_boards()
        if e > 0 and e % batch_size == 0:
            # Compute discounted rewards
            reward_board1 = compute_disc_rew(reward_board1)

            # Execute gradient discend
            optimizer.zero_grad()

            for i in range(len(reward_board1)):
                state = state_board1[i]
                action = action_board1[i]
                reward = action_board1[i]

                # Compute probs and sample action
                probs = policy_net(state)
                m = Categorical(probs)
                action = m.sample()

                # Compute -logprob and gradients
                loss = m.log_prob(action) * reward
                loss.backward()
            
            optimizer.step()

            state_board1 = torch.tensor([]).to(device)
            action_board1 = torch.tensor([]).to(device)
            reward_board1 = torch.tensor([]).to(device)

    torch.save(policy_net.state_dict(), './models/model_state_dict_4.pth')


if __name__ == "__main__":
    main()

#TODO: Allow the player2 to contribute to the training
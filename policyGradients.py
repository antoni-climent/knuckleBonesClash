import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from knuckleEnv import knuckle
from tqdm import tqdm
import random
import wandb
# from validation import validate

# Parameters
run = wandb.init(project='knuckle')
num_episodes = 1000000
batch_size = 4096
learning_rate = 0.001

# Save variables to wandb
run.config.update({"num_episodes": num_episodes, "batch_size": batch_size, "learning_rate": learning_rate})

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
    return (rewards - np.mean(rewards)) / np.std(rewards)
        
def random_move(env, state):
    # Generate random action untill a possible one is found
    while True:
        # Sample action
        action = random.randint(0,2)
        if env.is_valid_move(state[1], action):
            return action

def update_wins_losses(num_wins, num_losses, env):
    if env.calculate_score(env.board1) > env.calculate_score(env.board2):
        num_wins += 1
    else:
        num_losses += 1
    return num_wins, num_losses

def main():

    # History book
    state_board1 = []
    action_board1 = []
    reward_board1 = []
    
    env = knuckle()
    policy_net = PolicyNetwork()
    wandb.watch(policy_net)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    num_wins = 0
    num_losses = 0

    for e in tqdm(range(num_episodes)):
        
        state, reward, done = env.reset()
        

        while not done:
            ###### FIRST TURN ######
            # Prepare state and sample action
            state = np.append(np.array(state[0] + state[1]).flatten(), env.last_roll)
            state =(state - 3.5) / 1.71 # Normalize (calculated by hand)
            state = torch.tensor(state, dtype=torch.float, requires_grad=True).flatten()
            
            state_board1.append(state) # Save current state

            # Sample action
            probs = policy_net(state)
            m = Categorical(probs)
            action = m.sample()

            # Run step and save action and reward for the current state
            state, reward, done = env.step(action, 1)
            action_board1.append(action)
            reward_board1.append(reward)
            
            if done: # Track wins and losses
                num_wins, num_losses = update_wins_losses(num_wins, num_losses, env)
                

            ###### SECOND TURN ######
            action = random_move(env, state)
            # Run step and save action and reward for the current state
            state, reward, done = env.step(action, 2)

            if done: # Track wins and losses
                num_wins, num_losses = update_wins_losses(num_wins, num_losses, env)

        if e > 0 and e % batch_size == 0:
            # Compute discounted rewards
            reward_board1 = compute_disc_rew(reward_board1)

            # Execute gradient discend
            optimizer.zero_grad()

            total_loss = 0
            for i in range(len(reward_board1)):
                state = state_board1[i]
                action = action_board1[i]
                reward = action_board1[i]

                # Compute probs and sample action
                probs = policy_net(state)
                m = Categorical(probs)
                action = m.sample()

                # Compute -logprob and gradients
                loss = -m.log_prob(action) * reward
                total_loss += loss
                loss.backward()

            # Normalize the gradients to prevent exploding gradients
            for param in policy_net.parameters():
                if param.grad is not None:
                    param.grad /= len(reward_board1)

            # Log the loss    
            wandb.log({"loss": total_loss/len(reward_board1)})

            # Update the weights
            optimizer.step()

            state_board1 = []
            action_board1 = []
            reward_board1 = []

        if e % 100000 == 0:
            # validate(policy_net)
            torch.save(policy_net.state_dict(), f'./models/model_state_dict_{e}.pth')
        if e % 10000 == 0:
            run.log({"num_wins": num_wins, "num_losses": num_losses, "win_rate": num_wins / (num_wins + num_losses + 0.001)})
            num_wins, num_losses = 0, 0
    torch.save(policy_net.state_dict(), './models/model_state_dict_final.pth')


if __name__ == "__main__":
    main()

#TODO: Allow the player2 to contribute to the training
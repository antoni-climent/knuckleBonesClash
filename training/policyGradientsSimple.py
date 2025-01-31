import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from knuckleEnv import knuckle
from tqdm import tqdm
import random
import wandb

learning_rate = 0.00001
num_epochs = 1000
num_episodes = 1024

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

def random_move(env, state):
    valid_moves = [action for action in range(3) if env.is_valid_move(state[1], action)]
    return random.choice(valid_moves)

def update_wins_losses(num_wins, num_losses, env):
    if env.calculate_score(env.board1) > env.calculate_score(env.board2):
        num_wins += 1
    else:
        num_losses += 1
    return num_wins, num_losses

def main(model_name=None):
    # Initialize wandb
    run = wandb.init(project='knuckle')
    # Save variables to wandb
    run.config.update({"num_episodes": num_episodes, "num_epochs": num_epochs, "learning_rate": learning_rate})

    env = knuckle()
    policy_net = PolicyNetwork()

    # Load model if it exists to continue training
    if model_name:
        policy_net.load_state_dict(torch.load(model_name))
        print(f'Model {model_name} loaded')
        
    wandb.watch(policy_net)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    num_wins = 0
    num_losses = 0
    total_loss = 0
    rewards1 = np.array([])
    rewards2 = np.array([])

    for epoch in tqdm(range(num_epochs)):
        for ep in range(num_episodes):
            
            state, reward, done = env.reset()
            
            while True:
                ###### FIRST TURN ######
                # Prepare state and sample action
                state = np.append(np.array(state[0] + state[1]).flatten(), env.last_roll)
                state =(state - 3.5) / 1.71 # Normalize (calculated by hand)
                state = torch.tensor(state, dtype=torch.float, requires_grad=True).flatten()

                # Sample action
                probs = policy_net(state)
                m = Categorical(probs)
                action = m.sample()

                # Run step and save action and reward for the current state
                state, reward, done = env.step(action, 1)
                rewards1 = np.append(rewards1, reward)

                # Normalize reward (calculated by hand)
                reward = (reward - 2.2)/4.0 # 36 is the maximum possible reward 

                # Compute -logprob and gradients
                loss = -m.log_prob(action) * reward
                loss.backward()

                # Update the weights
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                
                
                if done: # Track wins and losses
                    num_wins, num_losses = update_wins_losses(num_wins, num_losses, env)
                    break
                    

                ###### SECOND TURN ######
                action = random_move(env, state)
                # Run step and save action and reward for the current state
                state, reward, done = env.step(action, 2)
                rewards2 = np.append(rewards2, reward)

                if done: # Track wins and losses
                    num_wins, num_losses = update_wins_losses(num_wins, num_losses, env)
                    break
            
            

        # Log the loss    
        wandb.log({"loss": total_loss/num_episodes})
        total_loss = 0

        wandb.log({"reward1_mean": np.mean(rewards1), "reward2_mean": np.mean(rewards2)})
        wandb.log({"reward1_std": np.std(rewards1), "reward2_std": np.std(rewards2)})
        rewards1 = np.array([])
        rewards2 = np.array([])

        # Log the win rate
        run.log({"num_wins": num_wins, "num_losses": num_losses, "win_rate": num_wins / (num_wins + num_losses + 0.001)})
        num_wins, num_losses = 0, 0

        if epoch % 50 == 0:
            # validate(policy_net)
            torch.save(policy_net.state_dict(), f'./models/model_state_dict_{epoch + 2000}.pth')

    torch.save(policy_net.state_dict(), './models/model_state_dict_final.pth')


# if __name__ == "__main__":
#     main('./models/model2/model_state_dict_2000.pth')
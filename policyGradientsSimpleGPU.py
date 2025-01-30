import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from knuckleEnvGPU import knuckle
from tqdm import tqdm
import random
import wandb

# Parameters
run = wandb.init(project='knuckle')
learning_rate = 0.001
num_epochs = 500
num_episodes = 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Save variables to wandb
run.config.update({"num_episodes": num_episodes, "num_epochs": num_epochs, "learning_rate": learning_rate})

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
    env = knuckle()
    policy_net = PolicyNetwork()
    policy_net.to(device)

    if model_name:
        policy_net.load_state_dict(torch.load(model_name))
        print(f'Model {model_name} loaded')
    policy_net.to(device)
        
    wandb.watch(policy_net)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    num_wins = 0
    num_losses = 0
    total_loss = 0

    batch_size = 64  # Define the batch size

    for epoch in tqdm(range(num_epochs)):
        for _ in range(num_episodes // batch_size):
            # Initialize batch variables
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_dones = []

            # Generate a batch of episodes
            for _ in range(batch_size):
                episode_states = []
                episode_actions = []
                episode_rewards = []
                episode_dones = []

                state, reward, done = env.reset()
                
                while not done:
                    # First turn
                    state = torch.cat([state[0].flatten(), state[1].flatten(), torch.tensor(env.last_roll, dtype=torch.int32, device=device).flatten()]).to(device)
                    state = (state - 3.5) / 1.71  # Normalize

                    probs = policy_net(state)
                    m = Categorical(probs)
                    action = m.sample()

                    next_state, reward, done = env.step(action, 1)

                    episode_states.append(state)
                    episode_actions.append(action)
                    episode_rewards.append(reward / 36)  # Normalize reward
                    episode_dones.append(done)

                    if done:
                        num_wins, num_losses = update_wins_losses(num_wins, num_losses, env)
                        break

                    # Second turn (random move)
                    action = random_move(env, next_state)
                    state, reward, done = env.step(action, 2)

                    if done:
                        num_wins, num_losses = update_wins_losses(num_wins, num_losses, env)

                # Add episode data to batch
                batch_states.extend(episode_states)
                batch_actions.extend(episode_actions)
                batch_rewards.extend(episode_rewards)
                batch_dones.extend(episode_dones)

            # Convert batch data to tensors
            batch_states = torch.stack(batch_states)
            batch_actions = torch.stack(batch_actions)
            batch_rewards = torch.tensor(batch_rewards, device=device)
            batch_dones = torch.tensor(batch_dones, device=device)

            # Compute loss for the entire batch
            probs = policy_net(batch_states)
            m = Categorical(probs)
            log_probs = m.log_prob(batch_actions)
            loss = -(log_probs * batch_rewards).mean()

            # Backpropagate and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Log metrics
        wandb.log({"loss": total_loss / (num_episodes // batch_size)})
        total_loss = 0

        run.log({"num_wins": num_wins, "num_losses": num_losses, "win_rate": num_wins / (num_wins + num_losses + 0.001)})
        num_wins, num_losses = 0, 0

        if epoch % 20 == 0:
            torch.save(policy_net.state_dict(), f'./models/model_state_dict_{epoch + 600}.pth')

    torch.save(policy_net.state_dict(), './models/model_state_dict_final.pth')

if __name__ == "__main__":
    main('./models/model2/model_state_dict_600.pth')

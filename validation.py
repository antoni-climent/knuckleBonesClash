import torch 
from knuckleEnv import knuckle
from policyGradients import PolicyNetwork
import random
from torch.distributions import Categorical
from tqdm import tqdm
from colorama import init
from termcolor import colored
import numpy as np

init()

num_episodes = 1000  


# Make the model compeate against a random one
def validate(model):

    # Initialize environment
    env = knuckle()

    wins = 0
    for i in tqdm(range(num_episodes)):
        state, _, _ = env.reset()
        done = False
        while not done:
            ###### AI TURN ######
            # Prepare state and sample action
            state = np.append(np.array(state[0] + state[1]).flatten(), env.last_roll)
            state = torch.tensor(state, dtype=torch.float32, requires_grad=False)
            state = state / 6 # Normalize

            # Sample action
            probs = model(state)
            m = Categorical(probs)
            action = m.sample()

            # Run step and save action and reward for the current state
            state, reward, done = env.step(action, 1)

            if done:
                break

            ###### RANDOM TURN ######
            # Reverse board position to encode the state
            #state = torch.tensor(state[1] + state[0], dtype=torch.float, requires_grad=True).flatten().to(device)

            # Generate random action untill a possible one is found
            while True:
                # Sample action
                action = random.randint(0,2)
                if env.is_valid_move(state[1], action):
                    break


            # Run step and save action and reward for the current state
            state, reward, done = env.step(action, 2)
        if env.calculate_score(env.board1) > env.calculate_score(env.board2):
            wins += 1

    print(f"AI won {wins} out of {num_episodes} games ({wins/num_episodes*100}%)")

def play_against_ai(path_name):
    # Load model
    model = PolicyNetwork()
    model.load_state_dict(torch.load(path_name))

    env = knuckle()
    state, reward, done = env.reset()
    while not done:
        ###### YOUR TURN ######
        action = int(input(f"Got {env.last_roll}. Insert pos: ")) - 1

        state, reward, done = env.step(action, 1)
        if done:
            break

        ###### AI TURN ######
        # Prepare state and sample action
        state = np.append(np.array(state[1] + state[0]).flatten(), env.last_roll)
        state = torch.tensor(state, dtype=torch.float32, requires_grad=False)
        state = (state - 3.5) / 1.71

        probs = model(state)
        action = np.argmax(probs.detach().numpy())

        print(colored(f"Machine got {env.last_roll} and inserted {action+1}. Distribution: {probs}", 'green'))

        # Run step
        state, reward, done = env.step(action, 2)

        # Display boards
        env.display_boards()

    print(f"Total score: {env.calculate_score(env.board1)} VS {env.calculate_score(env.board2)}")
    print(f"Last reward: {reward}")


if __name__ == "__main__":
    # Load model
    model_name = "./models/model_state_dict_80.pth"
    # model = PolicyNetwork()
    # model.load_state_dict(torch.load(model_name))
    # validate(model)
    play_against_ai(model_name)

    
        
import random
import torch
from colorama import init
from termcolor import colored

init()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

class knuckle:
    def __init__(self):
        self.board1 = torch.zeros((3, 3), dtype=torch.int32, device=device)
        self.board2 = torch.zeros((3, 3), dtype=torch.int32, device=device)
        self.last_roll = 0
        self.roll_dice()

    def roll_dice(self):
        self.last_roll = random.randint(1, 6)

    def calculate_column_score(self, column):
        dice_counts = torch.bincount(column, minlength=7)[1:]
        return torch.sum(torch.arange(1, 7, device=device) * dice_counts * dice_counts)

    def display_boards(self):
        print("Board 1:")
        print(self.board1.cpu().numpy())
        print("\nBoard 2:")
        print(self.board2.cpu().numpy())

    def clear_column(self, board, column_index):
        column = board[:, column_index]
        cleared_column = column[column != self.last_roll]
        padded_column = torch.cat([cleared_column, torch.zeros(3 - len(cleared_column), dtype=torch.int32, device=device)])
        board[:, column_index] = padded_column
        return board

    def place_choice(self, board, column_index):
        if board[2, column_index] != 0:
            return False
        empty_index = (board[:, column_index] == 0).nonzero(as_tuple=True)[0][0]
        board[empty_index, column_index] = self.last_roll
        return True

    def is_valid_move(self, board, column_index):
        return board[2, column_index] == 0

    def reset(self):
        self.board1.zero_()
        self.board2.zero_()
        return [self.board1, self.board2], 0, False

    def calculate_score(self, board):
        return torch.sum(torch.stack([self.calculate_column_score(board[:, i]) for i in range(3)]))

    def step(self, action, board_n):
        score1 = self.calculate_score(self.board1)
        score2 = self.calculate_score(self.board2)

        if board_n == 1:
            if not self.place_choice(self.board1, action):
                return [self.board1, self.board2], 0, True

            if self.last_roll in self.board2[:, action]:
                self.board2 = self.clear_column(self.board2, action)

        elif board_n == 2:
            if not self.place_choice(self.board2, action):
                return [self.board1, self.board2], 0, True

            if self.last_roll in self.board1[:, action]:
                self.board1 = self.clear_column(self.board1, action)

        game_finished = torch.all(self.board1[2, :] != 0) or torch.all(self.board2[2, :] != 0)

        self.roll_dice()

        new_score1 = self.calculate_score(self.board1)
        new_score2 = self.calculate_score(self.board2)

        if board_n == 1:
            step_reward = (new_score1 - score1) + (score2 - new_score2)
        else:
            step_reward = (new_score2 - score2) + (score1 - new_score1)

        return [self.board1, self.board2], step_reward.item(), game_finished.item()

if __name__ == "__main__":
    env = knuckle()
    state, reward, done = env.reset()
    while not done:
        action = int(input(f"Got {env.last_roll}. Insert pos: ")) - 1
        state, reward, done = env.step(action, 1)
        print(f'Your reward is: {reward}')
        if done:
            break
        action = random.randint(0, 2)
        print(colored(f"Machine got {env.last_roll} and inserted {action+1}", 'green'))
        state, reward, done = env.step(action, 2)
        print(f'Machine reward is: {reward}')
        env.display_boards()
    print(f"Total score: {env.calculate_score(env.board1)} VS {env.calculate_score(env.board2)}")
    print(f"Last reward: {reward}")

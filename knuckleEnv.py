import random
from colorama import init
from termcolor import colored
init()

class knuckle:
    def __init__(self):
        self.board1 = [[0,0,0],
                       [0,0,0],
                       [0,0,0]]
    
        self.board2 = [[0,0,0],
                       [0,0,0],
                       [0,0,0]]
        
        self.last_roll = 0
        self.roll_dice()

    def roll_dice(self):
        """Rolls a dice and returns a number between 1 and 6."""
        self.last_roll = random.randint(1, 6)

    def calculate_column_score(self, column):
        """Calculates the score of a column by multiplying the equal dice numbers and adding the others"""
        dice_counts = [0,0,0,0,0,0]
        for dice in column:
            if dice != 0:
                dice_counts[dice-1] += 1
        # Multiply indexes by the dice counts
        # print("Dice column: ", column, " with count: ", dice_counts)
        return sum(dice*count*count if count != 0 else 0 for dice, count in enumerate(dice_counts, 1))

    def display_boards(self):
        """Displays the game board."""
        for i in range(3):
            for j in range(3):
                # Print with colorama
                print(colored(self.board1[i][j],'red'), end="|")
            print("\n------")
        print()
        for i in range(3):
            for j in range(3):
                # Print with colorama
                print(colored(self.board2[i][j],'green'), end="|")
            print("\n------")

    def clear_column(self, board, column_index):
        """Clears de dices from the column that have the same value as the roll"""
        column = list(zip(*board))[column_index]
        cleared_column = [dice for dice in column if dice != self.last_roll]
        # Extend the cleared column with zeros to maintain the column length
        cleared_column.extend([0] * (3 - len(cleared_column)))
        for i in range(3):
            board[i][column_index] = cleared_column[i]
        return board

    def place_choice(self, board, column_index):
        """Places a dice value in the specified column."""
        if board[2][column_index] != 0:
            return False # If it is an invalid choice
        for j in range(3):
            if board[j][column_index] == 0:
                board[j][column_index] = self.last_roll
                break
        return True # If it is a valid choice
    
    def is_valid_move(self, board, column_index):
        if board[2][column_index] != 0:
            return False # If it is an invalid choice
        else:
            return True

    def reset(self):
        self.board1 = [[0,0,0],
                       [0,0,0],
                       [0,0,0]]
    
        self.board2 = [[0,0,0],
                       [0,0,0],
                       [0,0,0]]
        return [self.board1, self.board2], 0, False # Next state, reward, done
    
    def calculate_score(self, board):
        return sum(self.calculate_column_score(column) for column in list(zip(*board)))
    
    def step(self, action, board_n):
        # Calculate board scores before playing
        score1 = self.calculate_score(self.board1)
        score2 = self.calculate_score(self.board2)

        # Update board 1
        if board_n == 1:
            # If it breaks rules, return
            if not self.place_choice(self.board1, action):
                return [self.board1, self.board2], -1, True
            
            # If roll match with other player column number, clear it
            if self.last_roll in [row[action] for row in self.board2]:
                self.board2 = self.clear_column(self.board2, action)

            # Check if the game has finished
            game_finished = all(self.board1[2][i] != 0 for i in range(3)) or all(self.board2[2][i] != 0 for i in range(3))

            
            # print(f"Score1: {score1}, calculate_new_score: {self.calculate_score(self.board1)}")
            # Reward is the value increment of your board plus the subtraction value of the enemy
            step_reward = (self.calculate_score(self.board1) - score1) + (score2 - self.calculate_score(self.board2)) - self.last_roll + 0.01
            
            # Roll dice for next round
            self.roll_dice()
            return [self.board1, self.board2], step_reward, game_finished # Next state, reward, done
        
        # Update board 2
        elif board_n == 2:
            # If it breaks rules, return
            if not self.place_choice(self.board2, action):
                return [self.board1, self.board2], -1, True
            
            # If roll match with other player column number, clear it
            if self.last_roll in [row[action] for row in self.board1]: # Check if column 
                self.board1 = self.clear_column(self.board1, action)

             # Check if the game has finished
            game_finished = all(self.board1[2][i] != 0 for i in range(3)) or all(self.board2[2][i] != 0 for i in range(3))

            
            step_reward = (self.calculate_score(self.board2) - score2) + (score1 - self.calculate_score(self.board1)) - self.last_roll+ 0.01
            # Roll dice for next round
            self.roll_dice()
            return [self.board1, self.board2], step_reward, game_finished # Next state, reward, done

if __name__ == "__main__":
    env = knuckle()
    state, reward, done = env.reset()
    while not done:
        action = int(input(f"Got {env.last_roll}. Insert pos: ")) - 1
        state, reward, done = env.step(action, 1)
        print(f'Your reward is: {reward}')
        if done:
            break
        action = random.randint(0,2)
        print(colored(f"Machine got {env.last_roll} and inserted {action+1}", 'green'))
        state, reward, done = env.step(action, 2)
        print(f'Machine reward is: {reward}')
        env.display_boards()
    print(f"Total score: {env.calculate_score(env.board1)} VS {env.calculate_score(env.board2)}")
    print(f"Last reward: {reward}")
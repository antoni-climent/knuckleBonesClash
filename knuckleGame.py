import random
from colorama import init
from termcolor import colored
init()

def roll_dice():
    """Rolls a dice and returns a number between 1 and 6."""
    return random.randint(1, 6)

def calculate_column_score(column):
    """Calculates the score of a column by multiplying the equal dice numbers and adding the others"""
    dice_counts = [0,0,0,0,0,0]
    for dice in column:
        if dice != 0:
            dice_counts[dice-1] += 1
    # Multiply indexes by the dice counts
    print("Dice column: ", column, " with count: ", dice_counts)
    return sum(dice*count*count if count != 0 else 0 for dice, count in enumerate(dice_counts, 1))

def display_board(board, color):
    """Displays the game board."""
    for i in range(3):
        for j in range(3):
            # Print with colorama
            print(colored(board[i][j],color), end="|")
        print("\n------")

def clear_column(board, column_index, roll):
    """Clears de dices from the column that have the same value as the roll"""
    column = list(zip(*board))[column_index]
    cleared_column = [dice for dice in column if dice != roll]
    # Extend the cleared column with zeros to maintain the column length
    cleared_column.extend([0] * (3 - len(cleared_column)))
    for i in range(3):
        board[i][column_index] = cleared_column[i]
    return board

def place_choice(board, column_index, dice_value):
    """Places a dice value in the specified column."""
    for j in range(3):
        if board[j][column_index] == 0:
            board[j][column_index] = dice_value
            break

def play_knucklebones():
    """Main function to play the Knucklebones game."""
    # Initialize boards
    player_board = [[0,0,0],
                    [0,0,0],
                    [0,0,0]]
    
    computer_board = [[0,0,0],
                      [0,0,0],
                      [0,0,0]]

    print("Welcome to Knucklebones!\n")
    
    while True:
        ########## PLAYER'S TURN ##########
        player_roll = roll_dice()
        print(colored("-----------------", 'blue'))
        print(colored(f"You rolled a {player_roll}.", 'green'))

        # Player chooses a column
        while True:
            try:
                player_choice = int(input("Choose:")) - 1
                if player_choice in [0, 1, 2]:
                    if player_board[2][player_choice] != 0:
                        print("Column is full. Choose another column.")
                    else:
                        break
                else:
                    print("Invalid choice. Choose 1, 2, or 3.")
            except ValueError:
                print("Invalid input. Enter a number between 1 and 3.")

        # Place the dice in the chosen column
        place_choice(player_board, player_choice, player_roll)

        # Clear computer's column if duplicate is found
        if player_roll in [row[player_choice] for row in computer_board]:
            print(colored(f"You matched the computer's {player_roll} in column {player_choice + 1}!", 'green'))
            computer_board = clear_column(computer_board, player_choice, player_roll)
        
        if all(computer_board[2][i] != 0 for i in range(3)) or all(player_board[2][i] != 0 for i in range(3)):
            break
        ######## COMPUTER'S TURN ########

        computer_roll = roll_dice()
        # Computer chooses a column (simple strategy: random choice)
        while True:
            computer_choice = random.randint(0, 2)
            if computer_board[2][computer_choice] == 0: # Check if column is full
                break
            else:
                continue
        print(colored(f"The computer rolled a {computer_roll} and puts it in {computer_choice + 1}.", 'red'))
        place_choice(computer_board, computer_choice, computer_roll)

        # Clear player's column if duplicate is found
        if computer_roll in [row[computer_choice] for row in player_board]:
            print(colored(f"The computer matched your {computer_roll} in column {computer_choice + 1}! Clearing your column.", 'red'))
            player_board = clear_column(player_board, computer_choice, computer_roll)

        # Display the board
        print("Computer board:")
        display_board(computer_board, 'red')
        print("Your board:")
        display_board(player_board, 'green')

        # Check for win condition (one player has filled all columns)
        if all(computer_board[2][i] != 0 for i in range(3)) or all(player_board[2][i] != 0 for i in range(3)):
            break

    # Calculate final scores transposing boards
    print("Calculate player score")
    player_score = sum(calculate_column_score(column) for column in list(zip(*player_board)))
    print("Calculate computer score")
    computer_score = sum(calculate_column_score(column) for column in list(zip(*computer_board)))

    # Display the board
    print("Game Over!")
    print("Computer board:")
    display_board(computer_board, 'red')
    print("Your board:")
    display_board(player_board, 'green')
    print(f"Final Scores:\nPlayer: {player_score}\nComputer: {computer_score}")
    print("Player board:")
    print(player_board)
    print("Computer board:")
    print(computer_board)
    if player_score > computer_score:
        print("You win!")
    elif player_score < computer_score:
        print("The computer wins!")
    else:
        print("It's a tie!")

if __name__ == "__main__":
    play_knucklebones()

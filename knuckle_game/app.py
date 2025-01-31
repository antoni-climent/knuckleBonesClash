from flask import Flask, render_template, jsonify, request
import random

app = Flask(__name__)

class Knuckle:
    def __init__(self):
        self.board1 = [[0,0,0],[0,0,0],[0,0,0]]
        self.board2 = [[0,0,0],[0,0,0],[0,0,0]]
        self.last_roll = 0
        self.roll_dice()

    def roll_dice(self):
        self.last_roll = random.randint(1, 6)

    def place_choice(self, board, column_index):
        if board[2][column_index] != 0:
            return False
        for j in range(3):
            if board[j][column_index] == 0:
                board[j][column_index] = self.last_roll
                break
        return True

    def clear_column(self, board, column_index):
        column = list(zip(*board))[column_index]
        cleared_column = [dice for dice in column if dice != self.last_roll]
        cleared_column.extend([0] * (3 - len(cleared_column)))
        for i in range(3):
            board[i][column_index] = cleared_column[i]

    def step(self, action, board_n):
        if board_n == 1:
            if not self.place_choice(self.board1, action):
                return False
            if self.last_roll in [row[action] for row in self.board2]:
                self.clear_column(self.board2, action)
        else:
            if not self.place_choice(self.board2, action):
                return False
            if self.last_roll in [row[action] for row in self.board1]:
                self.clear_column(self.board1, action)
        self.roll_dice()
        return True

game = Knuckle()

@app.route('/')
def index():
    return render_template('index.html', game=game)

@app.route('/move', methods=['POST'])
def move():
    data = request.json
    column = data['column']
    if game.step(column, 1):
        ai_move = random.randint(0, 2)
        game.step(ai_move, 2)
        return jsonify({'success': True, 'last_roll': game.last_roll, 'ai_move': ai_move})
    return jsonify({'success': False})

@app.route('/reset', methods=['POST'])
def reset():
    global game
    game = Knuckle()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)

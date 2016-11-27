import random
from board import Board

class XInARow:
  def __init__(self, playerX, playerO, row, col, x_to_win):
    self.playerX, self.playerO = playerX, playerO
    self.playerX_turn = random.choice([True, False])
    self.board = Board(row, col, x_to_win)

  def play_game(self):
    self.playerX.start_game('X', self.board.state)
    self.playerO.start_game('O', self.board.state)
    while True:
      if self.playerX_turn:
        player = self.playerX
        char = 'X'
        other_player = self.playerO
      else:
        player = self.playerO
        char = 'O'
        other_player = self.playerX

      if player.breed == 'human':
        self.board.display_board()

      move = player.move(self.board)

      if move is None: # illegal move ONLY FOR REAL LIFE PLAYER
        player.reward(-99, self.board) # score of shame
        break

      self.board.move(move, char)

      if self.board.player_wins(char):
        player.reward(1, self.board)
        other_player.reward(-1, self.board)
        player.wins += 1
        break

      if self.board.board_full(): # tie game
        player.reward(0.5, self.board)
        other_player.reward(0.5, self.board)
        break

      other_player.reward(0, self.board)
      self.playerX_turn = not self.playerX_turn

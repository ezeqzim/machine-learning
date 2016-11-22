import random
from player import Player

class BoboPlayer(Player):
  def __init__(self):
    self.breed = 'bobo'
    self.wins = 0

  def reward(self, value, board):
    pass

  def start_game(self, char, board):
    pass

  def move(self, board):
    return board.available_moves_filtered()[len(board.available_moves_filtered())-1]

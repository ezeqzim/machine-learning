import random
from player import Player

class RandomPlayer(Player):
  def __init__(self):
    self.breed = 'random'
    self.wins = 0

  def reward(self, value, board):
    pass

  def start_game(self, char, board):
    pass

  def move(self, board):
    return random.choice(board.available_moves_filtered())

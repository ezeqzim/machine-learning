import random, sys, readline
import numpy as np
import matplotlib.pyplot as plt
from board import Board
from player import Player
from random_player import RandomPlayer
from q_learning_player import QLearningPlayer

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
        #player.reward(-99, self.board.state) # score of shame
        player.reward(-99, self.board)
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

def modo_de_uso():
  print 'Modo de uso:'
  print 'rows= cols= X= iter= mode='
  print 'Mode = qq, qr, rr, q, r, pp (q = q learning, r = random, p = player'
  print 'Si hay 1 q learning epsilon= alpha= gamma='
  print 'Si hay 2 q learning epsilon1= alpha1= gamma=1 epsilon2= alpha2= gamma2='
  sys.exit()

def main(**kwargs):
  try:
    rows = int(kwargs['rows'])
    cols = int(kwargs['cols'])
    x_to_win = int(kwargs['X'])
    play_mode = kwargs['mode']
  except:
    modo_de_uso()

  try:
    iterations = int(kwargs['iter'])
  except:
    modo_de_uso()

  if play_mode == 'pp':
    p1 = Player()
    p2 = Player()
  elif play_mode == 'q':
    p1 = Player()
    p2 = QLearningPlayer()
  elif play_mode == 'r':
    p1 = Player()
    p2 = RandomPlayer()
  elif play_mode == 'qq':
    try:
      epsilon1 = float(kwargs['epsilon1'])
    except:
      epsilon1 = 0.2
    try:
      alpha1 = float(kwargs['alpha1'])
    except:
      alpha1 = 0.3
    try:
      gamma1 = float(kwargs['gamma1'])
    except:
      gamma1 = 0.9
    try:
      epsilon2 = float(kwargs['epsilon2'])
    except:
      epsilon2 = 0.2
    try:
      alpha2 = float(kwargs['alpha2'])
    except:
      alpha2 = 0.3
    try:
      gamma2 = float(kwargs['gamma2'])
    except:
      gamma2 = 0.9
    p1 = QLearningPlayer(epsilon1, alpha1, gamma1)
    p2 = QLearningPlayer(epsilon2, alpha2, gamma2)
  elif play_mode == 'rq':
    try:
      epsilon = float(kwargs['epsilon'])
    except:
      epsilon = 0.2
    try:
      alpha = float(kwargs['alpha'])
    except:
      alpha = 0.3
    try:
      gamma = float(kwargs['gamma'])
    except:
      gamma = 0.9
    p1 = RandomPlayer()
    p2 = QLearningPlayer(epsilon, alpha, gamma)
  elif play_mode == 'rr':
    p1 = RandomPlayer()
    p2 = RandomPlayer()
  else:
    modo_de_uso()

  plotP1 = []
  plotP2 = []
  for i in xrange(1, iterations + 1):
    t = XInARow(p1, p2, rows, cols, x_to_win)
    t.play_game()
    # print >> sys.stderr, str(p1.wins) + '\t' + str(p2.wins) + '\t' + str(ties)
    window = 500.0
    if i % window == 0:
      percentage_p1 = 100 * float(p1.wins) / i
      percentage_p2 = 100 * float(p2.wins) / i
      plotP1.append(percentage_p1)
      plotP2.append(percentage_p2)
      print >> sys.stderr, str(percentage_p1) + '\t' + str(i)
      print >> sys.stderr, str(percentage_p2) + '\t' + str(i)
  p1 = Player()

  xax = np.arange(0.0, float(iterations), window)
  plt.plot(xax, plotP1, 'r-', xax, plotP2, 'b-')
  plt.show()

  try:
    p2.epsilon = 0
  except:
    pass

  while True:
    t = XInARow(p1, p2, rows, cols, x_to_win)
    t.play_game()

if __name__ == '__main__':
  kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if '=' in x)
  main(**kwargs)


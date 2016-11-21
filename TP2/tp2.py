import random, sys, readline
import numpy as np
import matplotlib.pyplot as plt
from x_in_a_row import XInARow
from player import Player
from random_player import RandomPlayer
from dumb_player import DumbPlayer
from q_learning_player import QLearningPlayer

def modo_de_uso():
  print 'Modo de uso:'
  print 'rows= cols= X= mode= iter= filename='
  print 'Mode = exp#'
  print 'Si hay 2 q learning epsilon1= alpha1= gamma=1 epsilon2= alpha2= gamma2='
  print 'Para jugar contra el Q player entrenado, manual=True'
  sys.exit()

def plotGraph(iterations, window, plotP1, plotP2, filename, label1, label2):
  xax = np.arange(0.0, float(iterations), window)
  plt.plot(xax, plotP1, 'r-', label=label1)
  plt.plot(xax, plotP2, 'b-', label=label2)
  plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
  plt.savefig(filename)
  plt.clf()

def run(rows, cols, x_to_win, iterations, p1, p2, filename):
  plotP1 = []
  plotP2 = []
  plotAcumP1 = []
  plotAcumP2 = []
  winsP1Acum = 0.0
  winsP2Acum = 0.0
  for i in xrange(1, iterations + 1):
    t = XInARow(p1, p2, rows, cols, x_to_win)
    t.play_game()
    window = 500.0
    if i % window == 0:
      percentage_p1 = 100 * float(p1.wins) / window
      percentage_p2 = 100 * float(p2.wins) / window
      plotP1.append(percentage_p1)
      plotP2.append(percentage_p2)
      print >> sys.stderr, str(percentage_p1) + '\t' + str(percentage_p2) + '\t' + str(i)
      winsP1Acum += p1.wins
      winsP2Acum += p2.wins
      plotAcumP1.append(100 * float(winsP1Acum) / i)
      plotAcumP2.append(100 * float(winsP2Acum) / i)
      p1.wins = 0
      p2.wins = 0

  plotGraph(iterations, window, plotP1, plotP2, filename, p1.__class__.__name__, p2.__class__.__name__)
  plotGraph(iterations, window, plotAcumP1, plotAcumP2, filename + '_acum', p1.__class__.__name__, p2.__class__.__name__)

  return [winsP1Acum, winsP2Acum]

def best_player(p1, p2, winsP1Acum, winsP2Acum):
  return p1 if winsP1Acum >= winsP2Acum else p2

def exp1(rows, cols, x_to_win, iterations, epsilon, alpha, gamma, filename):
  dumb_player = DumbPlayer()
  random_player = RandomPlayer()
  run(rows, cols, x_to_win, iterations, dumb_player, random_player, filename)
  return dumb_player

def exp2(rows, cols, x_to_win, iterations, epsilon1, alpha1, gamma1, epsilon2, alpha2, gamma2, filename):
  q_player1 = QLearningPlayer(epsilon1, alpha1, gamma1)
  q_player2 = QLearningPlayer(epsilon2, alpha2, gamma2)
  run(rows, cols, x_to_win, iterations, q_player1, q_player2, filename + '_training')
  q_player1.epsilon = 0
  q_player2.epsilon = 0
  random_player = RandomPlayer()
  statsQ1R = run(rows, cols, x_to_win, iterations, q_player1, random_player, filename + '_p1_test')
  statsQ2R = run(rows, cols, x_to_win, iterations, q_player2, random_player, filename + '_p2_test')
  return best_player(q_player1, q_player2, statsQ1R[0], statsQ2R[0])

def exp3(rows, cols, x_to_win, iterations, epsilon1, alpha1, gamma1, epsilon2, alpha2, gamma2, filename):
  q_player1 = QLearningPlayer(epsilon1, alpha1, gamma1)
  q_player2 = QLearningPlayer(epsilon2, alpha2, gamma2)
  statsQ1Q2 = run(rows, cols, x_to_win, iterations, q_player1, q_player2, filename + '_p1_p2_training')
  best_pl = best_player(q_player1, q_player2, statsQ1Q2[0], statsQ1Q2[1])
  best_pl.epsilon = 0
  q_player3 = QLearningPlayer()
  run(rows, cols, x_to_win, iterations, best_pl, q_player3, filename + '_best_p3_training')
  q_player3.epsilon = 0
  random_player = RandomPlayer()
  run(rows, cols, x_to_win, iterations, q_player3, random_player, filename + '_p3_random_test')
  return q_player3

def main(**kwargs):
  try:
    rows = int(kwargs['rows'])
    cols = int(kwargs['cols'])
    x_to_win = int(kwargs['X'])
    play_mode = kwargs['mode']
    iterations = int(kwargs['iter'])
    filename = kwargs['filename']
  except:
    modo_de_uso()

  if play_mode == '1':
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
    exp_player = exp1(rows, cols, x_to_win, iterations, epsilon, alpha, gamma, filename)
  elif play_mode == '2' or play_mode == '3':
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
    if(play_mode == '2'):
      exp_player = exp2(rows, cols, x_to_win, iterations, epsilon1, alpha1, gamma1, epsilon2, alpha2, gamma2, filename)
    else:
      exp_player = exp3(rows, cols, x_to_win, iterations, epsilon1, alpha1, gamma1, epsilon2, alpha2, gamma2, filename)
  else:
    modo_de_uso()

  try:
    kwargs['manual']
    player = Player()
    exp_player.epsilon = 0
    while True:
      t = XInARow(player, exp_player, rows, cols, x_to_win)
      t.play_game()
  except:
    pass

if __name__ == '__main__':
  kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if '=' in x)
  main(**kwargs)


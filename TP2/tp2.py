import random, copy, sys

class Board:
  def __init__(self, row, col, x_to_win):
    self.row = row
    self.col = col
    self.x_to_win = x_to_win
    self.available_moves = [0] * col
    self.state = [' '] * (col * row)

  def move(self, col, char):
    self.state[self.transform_move(self.available_moves[col], col)] = char
    self.last_move = (self.available_moves[col], col)
    self.available_moves[col] += 1

  def transform_move(self, row, col):
    return (self.col * row + col)

  def available_moves_filtered(self):
    return [(self.available_moves[i], i) for i in xrange(0, self.col) if self.available_moves[i] < self.row]

  def player_wins(self, char):
    row = self.last_move[0]
    col = self.last_move[1]
    char = self.state[self.transform_move(row, col)]
    # Horizontal Check
    count = 0
    for i in xrange(-self.x_to_win + 1, self.x_to_win):
      if not (col + i < 0 or col + i >= self.col):
        if self.state[self.transform_move(row, col + i)] == char:
          count += 1
        else:
          count = 0
        if count == self.x_to_win:
          return True

    # Vertical Check
    count = 0
    for i in xrange(-self.x_to_win + 1, self.x_to_win):
      if not (row + i < 0 or row + i >= self.row):
        if self.state[self.transform_move(row + i, col)] == char:
          count += 1
        else:
          count = 0
        if count == self.x_to_win:
          return True

    # Down Diagonal Check
    count = 0
    for i in xrange(-self.x_to_win + 1, self.x_to_win):
      if ((not (row - i < 0 or row - i >= self.row)) and (not (col + i < 0 or col + i >= self.col))):
        if self.state[self.transform_move(row - i, col + i)] == char:
          count += 1
        else:
          count = 0
        if count == self.x_to_win:
          return True

    # Up Diagonal Check
    count = 0
    for i in xrange(-self.x_to_win + 1, self.x_to_win):
      if ((not (row + i < 0 or row + i >= self.row)) and (not (col + i < 0 or col + i >= self.col))):
        if self.state[self.transform_move(row + i, col + i)] == char:
          count += 1
        else:
          count = 0
        if count == self.x_to_win:
          return True

    return False

  def board_full(self):
    return len(self.available_moves_filtered()) == 0

  def display_board(self):
    row = '| {} |'
    hr = '-----'
    print ((row * self.col + '\n' + hr * self.col + '\n') * self.row).format(*reversed(self.state))

  def illegal_move(self, col):
    bad = col < 0 or col >= self.col
    return bad or self.available_moves[col] == self.row

class XInARow:
  def __init__(self, playerX, playerO, row, col, x_to_win):
    self.playerX, self.playerO = playerX, playerO
    self.playerX_turn = random.choice([True, False])
    self.board = Board(row, col, x_to_win)

  def play_game(self):
    self.playerX.start_game('X', self.board)
    self.playerO.start_game('O', self.board)
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

      move = player.move(self.board) #(row, col)

      if move is None: # illegal move ONLY FOR REAL LIFE PLAYER
        player.reward(-99, self.board) # score of shame
        break

      self.board.move(move[1], char)

      if self.board.player_wins(char):
        player.reward(1, self.board)
        other_player.reward(-1, self.board)
        break

      if self.board.board_full(): # tie game
        player.reward(0.5, self.board)
        other_player.reward(0.5, self.board)
        break

      other_player.reward(0, self.board)
      self.playerX_turn = not self.playerX_turn

class Player(object):
  def __init__(self):
    self.breed = 'human'

  def start_game(self, char, board):
    print '\nNew game!'

  def move(self, board):
    col = int(raw_input('Your move (1 to {})? '.format(board.col)))
    col = board.col - col
    if board.illegal_move(col):
      return None
    return (board.available_moves[col], col)

  def reward(self, value, board):
    print '{} rewarded: {}'.format(self.breed, value)

class RandomPlayer(Player):
  def __init__(self):
    self.breed = 'random'

  def reward(self, value, board):
    pass

  def start_game(self, char, board):
    pass

  def move(self, board):
    return random.choice(board.available_moves_filtered())

class QLearningPlayer(Player):
  def __init__(self, epsilon=0.2, alpha=0.3, gamma=0.9):
    self.breed = 'Qlearner'
    self.harm_humans = True # Fuck Asimov
    self.q = {} # (state, action) keys: Q values
    self.epsilon = epsilon # e-greedy chance of random exploration
    self.alpha = alpha # learning rate
    self.gamma = gamma # discount factor for future rewards

  def start_game(self, char, board):
    self.last_board = copy.deepcopy(board)
    self.last_move = None

  def getQ(self, state, action):
    # encourage exploration; 'optimistic' 1.0 initial values
    if self.q.get((tuple(state), self.last_board.transform_move(action[0], action[1]))) is None:
        self.q[(tuple(state), self.last_board.transform_move(action[0], action[1]))] = 1.0 # Esto se puede modificar!!
    return self.q.get((tuple(state), self.last_board.transform_move(action[0], action[1])))

  def move(self, board):
    self.last_board = copy.deepcopy(board)
    actions = board.available_moves_filtered()

    if random.random() < self.epsilon: # explore!
      self.last_move = random.choice(actions)
      return self.last_move

    qs = [self.getQ(self.last_board.state, a) for a in actions]
    maxQ = max(qs)

    if qs.count(maxQ) > 1:
      # more than 1 best option; choose among them randomly
      best_options = [i for i in range(len(actions)) if qs[i] == maxQ]
      i = random.choice(best_options)
    else:
      i = qs.index(maxQ)

    self.last_move = actions[i]
    return actions[i]

  def reward(self, value, board):
    if self.last_move:
      self.learn(self.last_board, self.last_move, value, board)

  def learn(self, board, action, reward, board_result):
    prev = self.getQ(board.state, action)
    qs = [self.getQ(board_result.state, a) for a in board_result.available_moves_filtered()]
    if not len(qs) == 0:
      maxqnew = max(qs)
      self.q[(tuple(board.state), action)] = prev + self.alpha * (reward + self.gamma * maxqnew - prev)

def modo_de_uso():
  print "Modo de uso:"
  print "Los parametros requeridos son #Rows, #Cols, X in a Row to win. Luego"
  print "1. qq para 2 q learning, indicar #iteraciones. Optional epsilon, alpha, gamma"
  print "2. qr para 1 q learning y un random, indicar #iteraciones. Optional epsilon, alpha, gamma"
  print "3. rr para 2 random, indicar #iteraciones"
  print "4. q para 1 q learning y 1 player"
  print "5. r para 1 random y 1 player"
  print "6. pp para 2 player"
  sys.exit()

def main():
  if len(sys.argv) < 5:
    modo_de_uso()
  else:
    rows = int(sys.argv[1])
    cols = int(sys.argv[2])
    x_to_win = int(sys.argv[3])

    if sys.argv[4] == 'pp':
      p1 = Player()
      p2 = Player()
    elif sys.argv[4] == 'q':
      p1 = Player()
      p2 = QLearningPlayer()
    elif sys.argv[4] == 'r':
      p1 = Player()
      p2 = RandomPlayer()

    elif len(sys.argv) < 6:
      modo_de_uso()

    else:
      if len(sys.argv) > 6:
        epsilon = sys.argv[6]
      if len(sys.argv) > 7:
        alpha = sys.argv[7]
      if len(sys.argv) > 8:
        gamma = sys.argv[8]

      if sys.argv[4] == 'qq':
        p1 = QLearningPlayer()
        p2 = QLearningPlayer()
      elif sys.argv[4] == 'qr':
        p1 = QLearningPlayer()
        p2 = RandomPlayer()
      elif sys.argv[4] == 'rr':
        p1 = RandomPlayer()
        p2 = RandomPlayer()
      else:
        modo_de_uso()

      iterations = int(sys.argv[5])
      for i in xrange(0, iterations):
        t = XInARow(p1, p2, rows, cols, x_to_win)
        t.play_game()

      p1 = Player()
      p2.epsilon = 0

    while True:
      t = XInARow(p1, p2, rows, cols, x_to_win)
      t.play_game()

if __name__ == '__main__':
  main()


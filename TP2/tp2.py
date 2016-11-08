import random, copy, sys, readline

global wins_p1
global wins_p2
global ties

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
    global wins_p1
    global wins_p2
    global ties

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
        player.reward(-99, self.board.state) # score of shame
        break

      self.board.move(move[1], char)

      if self.board.player_wins(char):
        player.reward(10, self.board.state)
        other_player.reward(-20, self.board.state)
        if self.playerX_turn:
          wins_p1 += 1
        else:
          wins_p2 += 1
        break

      if self.board.board_full(): # tie game
        player.reward(0.5, self.board.state)
        other_player.reward(0.5, self.board.state)
        ties += 1
        break

      other_player.reward(0, self.board.state)
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
    # self.last_board = copy.deepcopy(board)
    self.last_board = board
    self.last_move = None

  def getQ(self, state, action):
    # encourage exploration; 'optimistic' 1.0 initial values
    if self.q.get((tuple(state), self.last_board.transform_move(action[0], action[1]))) is None:
        self.q[(tuple(state), self.last_board.transform_move(action[0], action[1]))] = 1.0 # Esto se puede modificar!!
    return self.q.get((tuple(state), self.last_board.transform_move(action[0], action[1])))

  def move(self, board):
    # self.last_board = copy.deepcopy(board)
    self.last_board = board
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

  def reward(self, value, state):
    if self.last_move:
      self.learn(value, state)

  def learn(self, reward, result_state):
    prev = self.getQ(self.last_board.state, self.last_move)
    qs = [self.getQ(result_state, a) for a in self.last_board.available_moves_filtered()]
    if len(qs) > 0:
      maxqnew = max(qs)
      self.q[(tuple(self.last_board.state), self.last_move)] = prev + self.alpha * (reward + self.gamma * maxqnew - prev)

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

  global wins_p1
  wins_p1 = 0
  global wins_p2
  wins_p2 = 0
  global ties
  ties = 0

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

  for i in xrange(0, iterations):
    t = XInARow(p1, p2, rows, cols, x_to_win)
    t.play_game()
    # print >> sys.stderr, str(wins_p1) + '\t' + str(wins_p2) + '\t' + str(ties)
    print >> str(wins_p2 - wins_p1)

  p1 = Player()
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


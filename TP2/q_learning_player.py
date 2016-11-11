import copy, random
from player import Player

class QLearningPlayer(Player):
  def __init__(self, epsilon=0.2, alpha=0.3, gamma=0.9):
    self.breed = 'Qlearner'
    self.harm_humans = True # Fuck Asimov
    self.q = {} # (state, action) keys: Q values
    self.epsilon = epsilon # e-greedy chance of random exploration
    self.alpha = alpha # learning rate
    self.gamma = gamma # discount factor for future rewards
    self.wins = 0

  def start_game(self, char, state):
    self.last_state = copy.deepcopy(state)
    self.last_move = None

  def getQ(self, state, action):
    # encourage exploration; 'optimistic' 1.0 initial values
    if self.q.get((tuple(state), action)) is None:
        self.q[(tuple(state), action)] = 1.0 # Esto se puede modificar!!
    return self.q.get((tuple(state), action))

  def move(self, board):
    self.last_state = copy.deepcopy(board.state)
    actions = board.available_moves_filtered()

    if random.random() < self.epsilon: # explore!
      self.last_move = random.choice(actions)
      return self.last_move

    qs = [self.getQ(self.last_state, action) for action in actions]
    maxQ = max(qs)

    if qs.count(maxQ) > 1:
      # more than 1 best option; choose among them randomly
      i = random.choice([i for i in range(len(actions)) if qs[i] == maxQ])
    else:
      i = qs.index(maxQ)

    self.last_move = actions[i]
    return actions[i]

  def reward(self, value, board):
    if self.last_move:
      self.learn(value, board)

  def learn(self, reward, result_board):
    prev = self.getQ(self.last_state, self.last_move)
    try:
      maxqnew = max([self.getQ(result_board.state, action) for action in result_board.available_moves_filtered()])
      self.q[(tuple(self.last_state), self.last_move)] = prev + self.alpha * (float(reward) + self.gamma * maxqnew - prev)
    except:
      pass

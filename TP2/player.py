class Player(object):
  def __init__(self):
    self.breed = 'human'
    self.wins = 0

  def start_game(self, char, board):
    print '\nNew game!'

  def move(self, board):
    col = int(raw_input('Your move (1 to {})? '.format(board.col)))
    col = board.col - col
    if board.illegal_move(col):
      return None
    return col

  def reward(self, value, board):
    print '{} rewarded: {}'.format(self.breed, value)


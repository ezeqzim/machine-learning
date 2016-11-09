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
    return self.col * row + col

  def available_moves_filtered(self):
    return [i for i in xrange(0, self.col) if self.available_moves[i] < self.row]

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
    return not any(self.available_moves_filtered())

  def display_board(self):
    row = '| {} |'
    hr = '-----'
    print ((row * self.col + '\n' + hr * self.col + '\n') * self.row).format(*reversed(self.state))

  def illegal_move(self, col):
    bad = col < 0 or col >= self.col
    return bad or self.available_moves[col] == self.row

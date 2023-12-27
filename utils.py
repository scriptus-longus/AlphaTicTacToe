import torch
import env
import random

class GameMem:
  def __init__(self):
    self.player_states = []
    self.actions = []
    self.pis = []
    self.winner = None

  def add(self, state, player, pi, action):
    self.player_states.append((state, player))
    self.actions.append(action)
    self.pis.append(pi)

  def unpack(self):
    states = []
    values = []
  
    for s, p in self.player_states:
      states.append(s*p) 
      values.append(torch.tensor([self.winner*p]))

    return torch.cat(states).reshape(-1, 9), torch.cat(self.pis).reshape(-1, 9), torch.cat(self.actions).reshape(-1, 1), torch.cat(values).reshape(-1, 1).float()

class Mem:
  def __init__(self, capacity):
    self.mem = []
    self.capacity = capacity

  def add(self, game):
    if len(self.mem) == self.capacity:
      #idx = random.randint(0, self.capacity)
      self.mem.pop(0)
      #self.mem[idx] = game
    self.mem.append(game)

  def get_batch(self, n_games, max_len=None):
    games = random.sample(self.mem, k=n_games)
    states = []
    actions = []
    zs = []
    pis = []
 
    for g in games:
      state, pi, a, z = g.unpack()
      states.append(state)
      pis.append(pi)
      actions.append(a)
      zs.append(z)

    return torch.cat(states).reshape(-1, 9), torch.cat(pis).reshape(-1, 9), torch.cat(actions).reshape(-1, 1), torch.cat(zs).reshape(-1, 1) 

def eval(model, n_games):
    model_wins = 1
    op_wins = 1

    for _ in range(n_games):
        game = env.TicTacToe()
        while not game.is_over():
            if game.player == 1:
                pi = torch.randn(9)
            else:
                pi, _ = model((game.state()*game.player).view(1, -1))

            pi = pi.masked_fill(game.get_moves() == 0, value=float("-inf"))
            pi = torch.softmax(pi, dim=-1)
            a = torch.multinomial(pi, 1)

            game.step(a.item())
        winner = game.get_winner()
        if winner == 1:
            op_wins += 1
        if winner == -1:
            model_wins += 1
    return op_wins/model_wins, op_wins, model_wins

def visits2pi(visits, tau=1):
  pi = torch.zeros(9)
  for a in visits:
    pi[a] = (visits[a]**(1/tau))/sum(x**(1/tau) for x in visits.values())

  return pi

def board2str(board):
  col = []

  for y in board:
    row = []
    for x in y:
      if (x == 1):
        row.append("x")
      elif (x == -1):
        row.append("o")
      else:
        row.append(" ")

    row_str = "|".join(row)
    col.append(row_str)
  ret = "\n-----\n".join(col)
  return ret

def print_tree(root, layer = 0, level = 0):
    if root.children == {}:
        return
        
    for c in root.children:
        if level == layer:
            print("state")
            if None != root.children[c].state:
                print(utils.board2str(root.children[c].state))
            else:
                print("None")
            print(f"prob: {root.children[c].prob.item()}")
            print(f"n wins: {root.children[c].parent_wins}")
            print("\n")
            
        print_tree(root.children[c], layer=layer, level=level+1)

    if level == layer:
        print("==================")

def print_game(game):
  player_decoder = {1: "x", -1: "o", 0: "None"}
  for i, (s, p) in enumerate(game.player_states):
    print(board2str(s))
    
    #player = "x" if p == 1 else "o"
    player = player_decoder[p]
    action = game.actions[i].item()

    print(f"player {player} about to move: {action}")  

  
  winner = player_decoder[game.winner]
  print(f"winner is {winner}") 

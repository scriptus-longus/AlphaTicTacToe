import torch
import env
import utils

class AlphaTTT(torch.nn.Module):
  def __init__(self):
    super(AlphaTTT, self).__init__()
    
    self.proj = torch.nn.Linear(3*3, 3*3*30)
    self.hid = torch.nn.Linear(3*3*30, 3*3*30)
    self.pol_head = torch.nn.Linear(3*3*30, 3*3)
    self.val_head = torch.nn.Linear(3*3*30, 1)

    self.act = torch.nn.ReLU()

  def forward(self, state):
    x = self.act(self.proj(state))
    x = self.act(self.hid(x))

    pi = self.pol_head(x)
    val = self.val_head(x)
    return pi, val

  def run(self, human_first=True):
    game = env.TicTacToe()
    
    if human_first:
      human_player = game.player
    else:
      human_player = game.player * -1

    while not game.is_over():
        print(utils.board2str(game.board))
        if game.player == human_player:
          while True:
            print("Your move: ")
            action = input()
            legal_moves = game.get_moves()
            if legal_moves[int(action)] == 1:
              break

            print("not a valid action")
        else:
          print("Network makes move...")
          pi, _ = self((game.state()*game.player).view(-1, 9))
          legal_moves = game.get_moves()

          pi = pi.masked_fill(legal_moves == 0, value=float("-inf"))
          pi = torch.softmax(pi, dim=-1).flatten()
          action = torch.multinomial(pi, 1).item()
 
        game.step(int(action))
        
        if game.is_over():
            break
    winner = game.get_winner() 
    if winner == human_player:
      print("human has won the game")
    if winner == human_player *-1:
      print("AI has won")
    if winner == 0:
      print("its a tie")

if __name__ == "__main__":
  model = AlphaTTT()
  model.run()

import torch
import env
import utils
import models
import tqdm
import math
import mcts

def self_play(model, n_games, n_simulations=1000, show_progress=True, debug=False):
    games = []

    if show_progress:
        it = tqdm.tqdm(range(n_games))
    else:
        it = range(n_games)
        
    
    for _ in it:
        game = env.TicTacToe()
        if debug:
            print("\n\nNew game.....")

        game_mem = utils.GameMem()
        while not game.is_over():
            with torch.no_grad():
            
                visits = mcts.mcts_search(model, game.board, game.player, n_simulations = n_simulations)
                pi = utils.visits2pi(visits)

                a = torch.multinomial(pi, 1)

                game_mem.add(torch.clone(game.board), game.player, torch.clone(pi), torch.clone(a))

                if debug:
                    print(utils.board2str(game.board))

                    player = "x" if game.player == 1 else "o"
                    action = a.item()
            
                    print(f"player {player} performs action {action}")
                game.step(a.item())
                
        game_mem.winner = game.get_winner()
        games.append(game_mem)
    return games

def train(model, optimizer, memory, bs=10, iters=10000, show_progress=True, eval_after=True):
  if show_progress:
    it = tqdm.tqdm(range(iters))
  else:
    it = range(iters)
  
  collected_losses = []
  for b in it:
    states, pis, a, zs = memory.get_batch(bs)
    B, C = states.shape
    
    ps, vs = model(states)
    
    loss = 0
    loss +=  torch.nn.functional.mse_loss(vs, zs)
    loss -= (pis * torch.log_softmax(ps, dim=-1)).mean()

    collected_losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  eval_score = None
  if eval_after:
    eval_score, op_wins, net_wins = utils.eval(model, 600)
  return collected_losses, eval_score
   

if __name__ == "__main__":
  mem = utils.Mem(300)
  model = models.AlphaTTT()
  epochs = 5

  optimizer = torch.optim.Adam(model.parameters())

  # load some starting samples into memory
  print("Initializing memory with a few rounds of self play...")
  games = self_play(model, 35, n_simulations=350, show_progress=True, debug=False)
  for g in games:
    mem.add(g)

  print("training")
  for e in range(epochs): 
    games = self_play(model, 25, n_simulations=350, show_progress=True, debug=False)
    for g in games:
      mem.add(g)

    losses, eval_score = train(model, optimizer, mem, bs=2, show_progress=True)
    print("avg train loss: ", sum(losses)/len(losses))
    print("eval score: ", eval_score)
    

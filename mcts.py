import torch
import random
import env
import math

class Node:
    def __init__(self, prob, player):
        self.visit_count = 0
        self.parent_wins = 0
        self.prob = prob
        self.player = player
        self.state = None
        self.children = {}

    def is_leaf(self):
        return self.children == {}
        

    def expand(self, pi):
        for i, prob in enumerate(pi):
            if prob != 0:
                self.children[i] = Node(prob, self.player*-1)
    
    def select(self):
        ucbs = {}
        for a, c in zip(self.children.keys(), self.children.values()):
            U = 1.25 * c.prob * math.sqrt(self.visit_count)/(c.visit_count+1)
            Q = c.parent_wins / (c.visit_count+1)
            ucbs[a] = Q + U

        a = max(ucbs, key=ucbs.get)
        child = self.children[a]
        return a, child

def mcts_search(model, state, player, n_simulations=1000):
    visits = {i: 0 for i in range(9)}
    root = Node(1, player)

    for _ in range(n_simulations):
        game = env.TicTacToe()
        game.board = torch.clone(state)
        game.player = player

        first_action = None
        path = [root]
        current_node = root

        # walk down tree
        while not current_node.is_leaf():
            a, current_node = current_node.select()
            path.append(current_node)
            game.step(a)
            current_node.state = torch.clone(game.board)

        # expand
        if not game.is_over():
            legal_moves = game.get_moves()
            pi, v = model((game.state().flatten()*game.player).view(1, -1))
            #pi = torch.randn(9)
            pi = pi.masked_fill(legal_moves == 0, value=float("-inf"))
            pi = torch.softmax(pi, dim=-1).flatten()
            

            current_node.expand(pi)
            # predicted playout
            winner = v.item()*game.player

            path.append(current_node)

        winner = winner if game.get_winner() == 0 else game.get_winner()
        path = list(reversed(path))

        #backprop
        if winner != 0:
            for i, n in enumerate(path[:-1]):
                n.visit_count += 1
                n.parent_wins += n.player*-1*winner
               

    for a in visits:
        if a in root.children.keys():
           visits[a] = root.children[a].visit_count
        
    return visits

def naive_search(model, state, player, n_simulations=1000):
    initial_actions = {i: 0 for i in range(9)}
    initial_player = player

    for _ in range(n_simulations):
        game = env.TicTacToe()
        game.board = torch.clone(state)
        game.player = player

        first_action = None
        
        while not game.is_over():
            legal_moves = game.get_moves()
            
            pi, v = model((game.state().flatten()*game.player).view(-1, 9))
            pi = pi.masked_fill(legal_moves == 0, value=float("-inf"))
            pi = torch.softmax(pi, dim=-1).flatten()
            a = torch.multinomial(pi, 1).flatten().item()

            if first_action == None:
                first_action = a

            game.step(a)

        winner = game.get_winner()
        
        if winner != 0:
            value = 1 if winner == initial_player else -1
        else:
            value = 0
            
        if first_action != None:
            initial_actions[first_action] += value

    return initial_actions

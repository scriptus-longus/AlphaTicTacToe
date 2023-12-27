import torch
import random

class TicTacToe:
    def __init__(self):
        self.player = random.choice([-1, 1])
        self.board = torch.zeros(3,3)
        
    def step(self, move):
        x = move//3
        y = move%3
        self.board[x][y] = self.player
        self.player *= -1
        
    def state(self):
        return self.board.flatten()
    
    def get_moves(self):
        return torch.zeros(3,3).masked_fill(self.board == 0, 1).flatten()
    
    def is_over(self):
        if (self.board[0] == 1).all() or (self.board[1] == 1).all() or (self.board[2] == 1).all():
            return True
        if (self.board[:, 0] == 1).all() or (self.board[:, 1] == 1).all() or (self.board[:, 2] == 1).all():
            return True
        if (self.board[0] == -1).all() or (self.board[1] == -1).all() or (self.board[2] == -1).all():
            return True
        if (self.board[:, 0] == -1).all() or (self.board[:, 1] == -1).all() or (self.board[:, 2] == -1).all():
            return True
        if (self.board.diag() == 1).all() or (self.board.flip(1).diag() == 1).all():
            return True
        if (self.board.diag() == -1).all() or (self.board.flip(1).diag() == -1).all():
            return True
        if (self.board != 0).all():
            return True
        return False
    
    def get_winner(self):
        if (self.board[0] == 1).all() or (self.board[1] == 1).all() or (self.board[2] == 1).all():
            return 1
        if (self.board[:, 0] == 1).all() or (self.board[:, 1] == 1).all() or (self.board[:, 2] == 1).all():
            return 1
        if (self.board[0] == -1).all() or (self.board[1] == -1).all() or (self.board[2] == -1).all():
            return -1
        if (self.board[:, 0] == -1).all() or (self.board[:, 1] == -1).all() or (self.board[:, 2] == -1).all():
            return -1
        if (self.board.diag() == 1).all() or (self.board.flip(1).diag() == 1).all():
            return 1
        if (self.board.diag() == -1).all() or (self.board.flip(1).diag() == -1).all():
            return -1
        return 0
            

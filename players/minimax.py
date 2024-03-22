class TTTMinimaxPlayer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.opponent_symbol = "O" if symbol == "X" else "X"

    def input(self, game):
        """Determine the best move using the Minimax algorithm."""
        _, best_move = self.minimax(game, True)
        return best_move

    def minimax(self, game, is_maximizing):
        """Minimax algorithm to evaluate the best move."""
        if game.check_win():
            return (1, None) if not is_maximizing else (-1, None)
        elif game.check_draw():
            return (0, None)

        best_move = None
        if is_maximizing:
            best_score = -float('inf')
            symbol = self.symbol
        else:
            best_score = float('inf')
            symbol = self.opponent_symbol

        for row in range(3):
            for col in range(3):
                if game.board[row][col] == " ":
                    game.board[row][col] = symbol
                    score, _ = self.minimax(game, not is_maximizing)
                    game.board[row][col] = " "
                    if is_maximizing and score > best_score:
                        best_score, best_move = score, (row, col)
                    elif not is_maximizing and score < best_score:
                        best_score, best_move = score, (row, col)
        return best_score, best_move


class TTTMinimaxABPPlayer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.opponent_symbol = "O" if symbol == "X" else "X"

    def input(self, game):
        """Determine the best move using the Minimax algorithm with Alpha-Beta Pruning."""
        _, best_move = self.minimax(game, True, -float('inf'), float('inf'))
        return best_move

    def minimax(self, game, is_maximizing, alpha, beta):
        """Minimax algorithm with Alpha-Beta Pruning to evaluate the best move."""
        if game.check_win():
            return (1, None) if not is_maximizing else (-1, None)
        elif game.check_draw():
            return (0, None)

        best_move = None
        if is_maximizing:
            best_score = -float('inf')
            symbol = self.symbol
        else:
            best_score = float('inf')
            symbol = self.opponent_symbol

        for row in range(3):
            for col in range(3):
                if game.board[row][col] == " ":
                    game.board[row][col] = symbol
                    score, _ = self.minimax(game, not is_maximizing, alpha, beta)
                    game.board[row][col] = " "
                    if is_maximizing:
                        if score > best_score:
                            best_score, best_move = score, (row, col)
                        alpha = max(alpha, score)
                    else:
                        if score < best_score:
                            best_score, best_move = score, (row, col)
                        beta = min(beta, score)
                    if beta <= alpha:
                        break
            if beta <= alpha:
                break
        return best_score, best_move

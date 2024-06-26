import random


class TTTMinimaxPlayer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.opponent_symbol = "O" if symbol == "X" else "X"

    def input(self, game):
        """Determine the best move using the Minimax algorithm."""
        if game.beginning:
            game.beginning = False
            return random.choice([(i, j) for i in range(3) for j in range(3)])  # Choose a random move
        else:
            _, best_move = self.minimax(game, True)
            return best_move

    @classmethod
    def to_string(cls) -> str:
        return "minimax"

    def minimax(self, game, is_maximizing):
        if game.check_win(self.symbol):  # Check if self.symbol has won
            return (1, None)
        elif game.check_win(self.opponent_symbol):  # Check if self.opponent_symbol has won
            return (-1, None)
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
        if game.beginning:
            game.beginning = False
            return random.choice([(i, j) for i in range(3) for j in range(3)])  # Choose a random move
        else:
            _, best_move = self.minimax(game, True, -float('inf'), float('inf'))
            return best_move

    @classmethod
    def to_string(cls) -> str:
        return "minimax_abp"

    def minimax(self, game, is_maximizing, alpha, beta):
        """Minimax algorithm with Alpha-Beta Pruning to evaluate the best move."""
        if game.check_win(self.symbol):  # Check if self.symbol has won
            return (1, None)
        elif game.check_win(self.opponent_symbol):  # Check if opponent has won
            return (-1, None)
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
                        if alpha >= beta:
                            break
                    else:
                        if score < best_score:
                            best_score, best_move = score, (row, col)
                        beta = min(beta, score)
                        if beta <= alpha:
                            break
        return best_score, best_move


class Connect4MinimaxPlayer:
    def __init__(self, symbol, max_depth=5):
        self.symbol = symbol
        self.opponent_symbol = "O" if symbol == "X" else "X"
        self.max_depth = max_depth

    def input(self, game):
        """Determine the best move using the Minimax algorithm."""
        if game.beginning:
            game.beginning = False
            return random.randint(0, 6)  # Choose a random column
        else:
            _, best_move = self.minimax(game, True, self.max_depth)
            return best_move

    @classmethod
    def to_string(cls) -> str:
        return "minimax"

    def minimax(self, game, is_maximizing, depth):
        if game.check_win(self.symbol):  # Check if self.symbol has won
            return (1, None)
        elif game.check_win(self.opponent_symbol):  # Check if self.opponent_symbol has won
            return (-1, None)
        elif game.check_draw():
            return (0, None)
        elif depth == 0:
            return (0, None)  # Return 0 score if reached max depth

        best_move = None
        if is_maximizing:
            best_score = -float('inf')
            symbol = self.symbol
        else:
            best_score = float('inf')
            symbol = self.opponent_symbol

        for col in range(7):
            if game.board[0][col] == " ":  # Check if the column is not full
                row = game.get_next_open_row(col)
                game.board[row][col] = symbol
                score, _ = self.minimax(game, not is_maximizing, depth - 1)
                game.board[row][col] = " "
                if is_maximizing and score > best_score:
                    best_score, best_move = score, col
                elif not is_maximizing and score < best_score:
                    best_score, best_move = score, col
        return best_score, best_move


class Connect4MinimaxABPPlayer:
    def __init__(self, symbol, max_depth=5):
        self.symbol = symbol
        self.opponent_symbol = "O" if symbol == "X" else "X"
        self.max_depth = max_depth

    def input(self, game):
        """Determine the best move using the Minimax algorithm with Alpha-Beta Pruning."""
        if game.beginning:
            game.beginning = False
            return random.randint(0, 6)  # Choose a random column
        else:
            _, best_move = self.minimax(game, True, self.max_depth, -float('inf'), float('inf'))
            return best_move

    @classmethod
    def to_string(cls) -> str:
        return "minimax_abp"

    def minimax(self, game, is_maximizing, depth, alpha, beta):
        """Minimax algorithm with Alpha-Beta Pruning to evaluate the best move."""
        if game.check_win(self.symbol):  # Check if self.symbol has won
            return (1, None)
        elif game.check_win(self.opponent_symbol):  # Check if opponent has won
            return (-1, None)
        elif game.check_draw():
            return (0, None)
        elif depth == 0:
            return (0, None)  # Return 0 score if reached max depth

        best_move = None
        if is_maximizing:
            best_score = -float('inf')
            symbol = self.symbol
        else:
            best_score = float('inf')
            symbol = self.opponent_symbol

        for col in range(7):
            if game.board[0][col] == " ":  # Check if the column is not full
                row = game.get_next_open_row(col)
                game.board[row][col] = symbol
                score, _ = self.minimax(game, not is_maximizing, depth - 1, alpha, beta)
                game.board[row][col] = " "
                if is_maximizing:
                    if score > best_score:
                        best_score, best_move = score, col
                    alpha = max(alpha, score)
                    if alpha >= beta:
                        break
                else:
                    if score < best_score:
                        best_score, best_move = score, col
                    beta = min(beta, score)
                    if beta <= alpha:
                        break
        return best_score, best_move

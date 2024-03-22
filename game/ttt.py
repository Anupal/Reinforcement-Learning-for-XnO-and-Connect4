class TicTacToe:
    def __init__(self):
        """Initialize the Tic Tac Toe board."""
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.current_player = "X"

    def print_board(self):
        """Prints the Tic Tac Toe board."""
        for row in self.board:
            print("|".join(row))
            print("-" * 5)

    def user_input(self, row, col):
        """Allows the player to place their mark on the board based on the given row and column."""
        if 0 <= row < 3 and 0 <= col < 3 and self.board[row][col] == " ":
            self.board[row][col] = self.current_player
            if self.check_win():
                return True, self.current_player
            elif self.check_draw():
                return True, None
            self.switch_player()
        else:
            print("Invalid move. Please try again.")
        return False, None

    def check_win(self):
        """Checks if the current player has won the game."""
        player = self.current_player
        for i in range(3):
            if all([self.board[i][j] == player for j in range(3)]) or \
               all([self.board[j][i] == player for j in range(3)]):
                return True
        if all([self.board[i][i] == player for i in range(3)]) or \
           all([self.board[i][2-i] == player for i in range(3)]):
            return True
        return False

    def check_draw(self):
        """Checks if the game is a draw."""
        return all([self.board[row][col] != " " for row in range(3) for col in range(3)])

    def switch_player(self):
        """Switches the turn to the other player."""
        self.current_player = "O" if self.current_player == "X" else "X"


def play_tic_tac_toe(player_x, player_o, display_board=True):
    game = TicTacToe()
    while True:
        if display_board:
            game.print_board()
        try:
            if game.current_player == "X":
                row, col = player_x.input(game)
            else:
                row, col = player_o.input(game)
            game_over, winner = game.user_input(row, col)
            if game_over:
                return winner

        except ValueError:
            print("Invalid input. Please enter numbers between 0 and 2.")

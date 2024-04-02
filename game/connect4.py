class Connect4:
    def __init__(self):
        """Initialize the Connect4 board."""
        self.rows = 6
        self.cols = 7
        self.board = [[" " for _ in range(self.cols)] for _ in range(self.rows)]
        self.current_player = "X"
        self.beginning = True

    @classmethod
    def to_string(cls) -> str:
        return "connect4"

    def print_board(self):
        """Prints the Connect4 board."""
        for row in self.board:
            print("|".join(row))
            print("-" * (4 * self.cols - 1))
        print()

    def user_input(self, col):
        """Allows the player to place their mark on the board based on the given column."""
        if self.is_valid_move(col):
            for row in range(self.rows - 1, -1, -1):
                if self.board[row][col] == " ":
                    self.board[row][col] = self.current_player
                    if self.check_win(self.current_player):  # Pass current player's symbol
                        return True, self.current_player
                    elif self.check_draw():
                        return True, None
                    self.switch_player()
                    return False, None
        else:
            print("Invalid move. Please try again.")
            return False, None
    
    def is_valid_move(self, col):
        """Checks if the given column is a valid move."""
        return 0 <= col < self.cols and self.board[0][col] == " "

    def get_next_open_row(self, col):
        """Find the next available row in a given column."""
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == " ":
                return row
        return -1  # Column is full

    def check_win(self, player_symbol):
        """Checks if the specified player has won the game."""
        for row in range(self.rows):
            for col in range(self.cols):
                if self.check_direction(player_symbol, row, col, 1, 0) or \
                   self.check_direction(player_symbol, row, col, 0, 1) or \
                   self.check_direction(player_symbol, row, col, 1, 1) or \
                   self.check_direction(player_symbol, row, col, 1, -1):
                    return True
        return False

    def check_direction(self, player_symbol, row, col, row_dir, col_dir):
        """Checks for a winning sequence in a given direction."""
        for i in range(4):
            if not (0 <= row + i * row_dir < self.rows and 0 <= col + i * col_dir < self.cols) or \
               self.board[row + i * row_dir][col + i * col_dir] != player_symbol:
                return False
        return True

    def check_draw(self):
        """Checks if the game is a draw."""
        return all([self.board[row][col] != " " for row in range(self.rows) for col in range(self.cols)])

    def switch_player(self):
        """Switches the turn to the other player."""
        self.current_player = "O" if self.current_player == "X" else "X"
    
    def get_possible_columns(self):
        """Returns a list of columns where a player can place their marker."""
        possible_columns = []
        for col in range(self.cols):
            if self.board[0][col] == " ":  # Check if the top row of the column is empty
                possible_columns.append(col)
        return possible_columns


def play_connect4(player_x, player_o, display_board=True):
    game = Connect4()
    while True:
        if display_board:
            print(f"{game.current_player}'s Turn")
            game.print_board()
        try:
            if game.current_player == "X":
                col = player_x.input(game)
            else:
                col = player_o.input(game)
            game_over, winner = game.user_input(col)
            if game_over:
                if display_board:
                    game.print_board()
                return winner
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 6.")

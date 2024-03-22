class TTTHumanPlayer:
    def __init__(self, label):
        self.label = label

    def input(self, game=None):
        row = int(input(f"Player {self.label}, enter your row (0-2): "))
        col = int(input(f"Player {self.label}, enter your column (0-2): "))

        return row, col
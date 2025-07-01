class Agent:
    def __init__(self, name="UnnamedAgent"):
        self.name = name

    def select_move(self, gamestate):
        """Given a GameState, return a valid move index (0-80)."""
        raise NotImplementedError("select_move must be implemented by subclasses")
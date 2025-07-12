# scripts/test_a_nn.py
from agents import get_agent
from engine.game import GameState
from engine.constants import EMPTY, X, O, PLAYER_MAP

short_games = [
    [55,3,18,72,54,9,28,12,27,0,10,30,2,24,56,15,29],
    [69,46,57,19,58,23,70,48,55,21,54,18,56,26,71,52,59],
    [69,45,54,20,70,49,57,19,58,22,59,26,71,52,55,21,56],
    [69,47,70,49,57,18,54,19,58,22,59,25,55,23,71,53,56],
    [69,46,57,19,59,25,58,21,54,20,70,48,55,22,56,24,71],
    [16,32,17,34,23,61,22,59,15,28,21,54,10,30,0,2,20],
    [20,62,16,32,17,33,10,31,22,59,15,28,21,54,0,61,23],
    [48,73,77,79,75,64,50,70,49,67,13,7,12,37,76,55,14],
    [48,73,75,64,40,49,77,70,32,16,14,34,13,43,76,4,12],
    [10,48,72,54,19,75,73,57,1,21,74,69,37,39,46,66,28],
    [74,69,46,75,73,57,19,66,37,48,72,9,28,3,1,21,10],
    [74,78,73,66,37,48,72,9,46,57,10,39,28,21,1,12,19],
    [74,78,73,75,72,57,19,66,46,30,10,48,37,39,28,13,1],
    [74,69,46,75,73,57,19,66,37,49,10,39,28,22,1,12,72],
    [64,30,10,39,47,69,29,6,0,9,38,42,73,57,20,78,55],
    [10,30,20,60,0,36,38,51,64,48,55,12,47,69,29,24,73],
    [6,20,62,16,32,17,33,0,1,23,60,10,31,22,59,15,28,21],
    [32,7,3,1,5,16,48,55,4,73,59,25,68,35,40,43,77],
    [32,7,3,1,5,25,59,16,40,31,4,64,48,55,68,52,77],
    [32,16,40,31,3,1,5,25,59,7,4,10,48,56,77,70,68],
    [32,15,29,17,42,36,28,12,27,10,40,39,33,9,51,65,48],
    [32,16,48,73,59,25,68,52,77,79,5,7,3,1,4,31,40],
    [32,7,5,25,59,16,40,31,3,10,48,73,68,52,77,62,4],
    [32,15,29,16,48,65,51,64,40,12,27,10,33,9,28,13,42],
    [32,25,59,16,48,55,5,7,3,19,68,52,77,62,40,73,4],
    [32,16,48,55,5,25,59,7,3,10,40,29,77,62,68,35,4],
    [32,16,48,74,78,64,40,54,10,3,18,20,80,70,79,57,2],
    [23,70,50,79,57,10,30,1,5,7,14,43,40,25,75,55,66],
    [23,79,66,28,5,16,50,61,14,52,57,10,40,49,75,73,30],
    [50,61,23,79,57,1,14,34,5,25,75,64,30,19,66,46,40],
    [50,61,5,25,57,19,75,73,66,28,23,70,30,1,14,34,40],
    [23,79,66,37,50,61,5,16,40,49,75,64,30,10,57,1,14],
    [14,43,50,79,75,55,5,7,23,70,40,31,30,19,57,10,66],
    [23,70,50,79,66,37,40,49,57,1,14,52,75,55,5,16,30],
    [23,79,57,19,66,46,75,55,14,43,30,10,50,70,40,37,5],
    [66,28,14,43,50,61,5,16,30,1,23,79,75,55,57,19,40],
    [66,46,57,19,75,73,14,52,40,49,23,70,30,10,50,79,5]
]

def test_winning_moves(agent):
    wrongs = 0
    for short_game in short_games:
        # Create a custom game state
        gs = GameState()

        # Simulate some moves to get a mid-game state (or edit board manually)
        # Example:
        for move in short_game[:-1]:
            gs.make_move(move)

        # Ask agent what it wants to do
        move = agent.select_move(gs)
        is_correct = move == short_game[-1]
        status = "✅" if is_correct else "❌"
        print(f"{status} Agent selected move: {move}, best move is {short_game[-1]}")
        if not is_correct:
            # Print current state
            print(f"(player to move is {PLAYER_MAP[gs.player]})")
            gs.print_board()
            wrongs+=1
    return wrongs

def test_losing_moves(agent):
    wrongs = 0
    for short_game in short_games:
        # Create a custom game state
        gs = GameState()

        # Simulate some moves to get a mid-game state (or edit board manually)
        # Example:
        for move in short_game[:-2]:
            gs.make_move(move)

        # Ask agent what it wants to do
        move = agent.select_move(gs)
        is_losing = move == short_game[-2]
        status = "✅" if not is_losing else "❌"
        print(f"{status} Agent selected move: {move}, worst move is {short_game[-1]}")
        if is_losing:
            # Print current state
            print(f"(player to move is {PLAYER_MAP[gs.player]})")
            gs.print_board()
            wrongs+=1
    return wrongs


def main():
    agent = get_agent("new_cnn")  # or any agent name in AGENT_FACTORIES

    # Optional: enable verbose mode on agent
    if hasattr(agent, 'verbose'):
        agent.verbose = False
    if hasattr(agent, 'set_eval'):
        agent.set_eval(True)

    w_wrongs = test_winning_moves(agent)
    l_wrongs = test_losing_moves(agent)
    
    print(f"Trying to immediate win, got it wrong: {w_wrongs}/{len(short_games)} = {100*w_wrongs/len(short_games):.1f}%")
    print(f"Trying to NOT immediate lose, got it wrong: {l_wrongs}/{len(short_games)} = {100*l_wrongs/len(short_games):.1f}%")
    print("Overall Results:")
    wrongs = w_wrongs + l_wrongs
    denom = len(short_games) * 2
    print(f"Mistakes: {wrongs}/{denom} = {100*wrongs/denom:.1f}%")


if __name__ == "__main__":
    main()

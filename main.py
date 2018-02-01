from agent import cls
from gomoku import GomokuEnv

from agent_pg import PGAgent
agent = PGAgent(board_size=9, win_len=5, hidden=500, model="models/pg_9_500.p")
#agent = PGAgent(board_size=3, win_len=3, model="models/pg_3_200.p")

'''from agent_torch import TorchAgent
agent = TorchAgent()'''

def run():
    cls()
    print("#" * 50 + "\n Welcome to Gomoku!\n By Tong Zou\n" + "#"*50 + "\n")
    while True:
        try:
            mode = eval(input('What would you like the Gomoku Agent to do?\n' + ' ' * 5 +
                         '1. Agent Training\n' + ' ' * 5 +
                         '2. Agent Vs Human\n' + ' ' * 5 +
                         '3. Human Vs Agent\n' + ' ' * 5 +
                         '4. Agent Vs Naive AI\n' + ' ' * 5 +
                         '5. Naive AI Vs Agent\n' + ' ' * 5 +
                         '6. Exit\n'))
            if mode < 1 or mode > 6:
                raise ValueError()
            break
        except:
            print("Please enter a valid choice.")

    if mode == 1:
        # play with a naive opponent
        agent.train(render=False, opponent='naive3', model_threshold=0.5)
        # self-play training
        #agent.train(render=False, model_threshold=0.4)
        input("Press Enter to continue...")
    elif mode == 2:
        agent.play(GomokuEnv.BLACK)
        input("Press Enter to continue...")
    elif mode == 3:
        agent.play(GomokuEnv.WHITE)
        input("Press Enter to continue...")
    elif mode == 4:
        agent.test(GomokuEnv.BLACK, 'naive3', render=True, size=10)
        input("Press Enter to continue...")
    elif mode == 5:
        agent.test(GomokuEnv.WHITE, 'naive3', render=True, size=10)
        input("Press Enter to continue...")
    else:
        exit()


if __name__ == '__main__':
    while True:
        run()

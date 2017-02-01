from agent import cls
from gomoku import GomokuEnv

import agent_pg
agent = agent_pg.PGAgent()

def run():
    cls()
    print "#" * 50 + "\n Welcome to Gomoku!\n By Tong Zou\n" + "#"*50 + "\n"
    while True:
        try:
            mode = input('What would you like the Gomoku Agent to do?\n' + ' ' * 5 +
                         '1. Self Train\n' + ' ' * 5 +
                         '2. Agent Vs Human\n' + ' ' * 5 +
                         '3. Human Vs Agent\n' + ' ' * 5 +
                         '4. Exit\n')
            if mode != 1 and mode != 2 and mode != 3 and mode != 4:
                raise ValueError()
            break
        except:
            print "Please enter a valid choice."

    if mode == 1:
        agent.learn(render=False)
    elif mode == 2:
        agent.play(GomokuEnv.BLACK)
        raw_input("Press Enter to continue...")
    elif mode == 3:
        agent.play(GomokuEnv.WHITE)
        raw_input("Press Enter to continue...")
    else:
        exit()


if __name__ == '__main__':
    while True:
        run()

from random import randrange
from ale_py import ALEInterface, SDL_SUPPORT
from ale_py.roms import Breakout

ale = ALEInterface()



# Get & Set the desired settings
ale.setInt("random_seed", 123)


# Check if we can display the screen
if SDL_SUPPORT:
    ale.setBool("sound", True)
    ale.setBool("display_screen", True)


ale.loadROM(Breakout)

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()

# Play 10 episodes
for episode in range(10):
    total_reward = 0
    while not ale.game_over():
        a = legal_actions[randrange(len(legal_actions))]
        # Apply an action and get the resulting reward
        reward = ale.act(a)
        total_reward += reward
    print("Episode %d ended with score: %d" % (episode, total_reward))
    ale.reset_game()
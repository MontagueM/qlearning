import abc


class AbstractGame(abc.ABC):
    @abc.abstractmethod
    def act(self):
        pass


class Agent(abc.ABC):
    def __init__(self, num_actions):
        self.num_actions = num_actions

    @abc.abstractmethod
    def act(self, game: 'AbstractGame'):
        pass

    @abc.abstractmethod
    def start_episode(self, episode_num):
        pass


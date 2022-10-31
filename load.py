import pickle

from training import run_game

agentName= "Task4QLearningWithDiscount"
with open(f"load/{agentName}", "rb") as pickle_file:
    agent = pickle.load(pickle_file)
    print(agent)
run_game(10, agent)

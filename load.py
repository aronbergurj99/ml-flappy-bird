import pickle

from training import run_game

agentName= "Task4Agent2"
with open(f"load/{agentName}", "rb") as pickle_file:
    agent = pickle.load(pickle_file)
    print(f"Agent trained for {agent.__dict__['frames_observed']} frames")
    print(f"Agent trained for {agent.__dict__['episodes_observed']} episodes")
print("Scores: ", run_game(10, agent, display_screen=False, force_fps=True))
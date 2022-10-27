
from numpy import mean
from ple.games.flappybird import FlappyBird
from ple import PLE
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
from pathlib import Path

from agents import *


def run_game(nb_episodes, agent, display_screen=True, force_fps=False):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    env = PLE(FlappyBird(), fps=30, display_screen=display_screen, force_fps=force_fps, rng=None,
            reward_values = reward_values)
    env.init()
    scores =[]
    score = 0
    while nb_episodes > 0:
        action = agent.policy(env.game.getGameState())
        reward = env.act(env.getActionSet()[action])
        score += reward
        if env.game_over():
            # print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            scores.append(score)
            score = 0

    return scores

def train(nb_episodes, agent: FlappyAgent, display_screen=False, force_fps=True):
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=display_screen, force_fps=force_fps, rng=None,
            reward_values = reward_values)
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        state = env.game.getGameState()
        action = agent.training_policy(state)

        # step the environment
        reward = env.act(env.getActionSet()[action])
        # print("reward=%d" % reward)

        # let the agent observe the current state transition
        newState = env.game.getGameState()
        agent.observe(state, action, reward, newState, env.game_over())

        score += reward
        # reset the environment if the game is over
        if env.game_over():
            # print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0


def observe_policy(agent, total_nb_episodes, numberOfObservations = 10):
    reward_values = agent.reward_values()
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    env.init()

    nb_episodes = total_nb_episodes//numberOfObservations
    score = 0
    
    now = datetime.now().strftime("%d-%m-%Y_%H %M %S")
    dirname = os.path.dirname(__file__)
    agentFolder = os.path.join(dirname, f"results/{now}")
    if not os.path.exists(agentFolder):
        os.makedirs(agentFolder)

    if not os.path.exists(agentFolder + "/plots"):
        os.makedirs(agentFolder + "/plots")

    for i in range(numberOfObservations):
        while nb_episodes > 0:
            # pick an action
            state = env.game.getGameState()
            action = agent.training_policy(state)

            # step the environment
            reward = env.act(env.getActionSet()[action])
            # print("reward=%d" % reward)

            # let the agent observe the current state transition
            newState = env.game.getGameState()
            agent.observe(state, action, reward, newState, env.game_over())

            score += reward
            # reset the environment if the game is over
            if env.game_over():
                # print("score for this episode: %d" % score)
                env.reset_game()
                nb_episodes -= 1
                score = 0

        nb_episodes = total_nb_episodes//numberOfObservations
        agent.plot("pi")
        agent.fig.savefig(agentFolder + f"/plots/heatmap{i + 1}")
    
    print("saving pickle")
    with open (agentFolder + "/" + agent.__class__.__name__, 'wb') as pickle_file:
        pickle.dump(agent, pickle_file)

if __name__ == "__main__":

    agent = TaskOneAgent()
    observe_policy(agent, 10000)


from email import header
import matplotlib
from numpy import mean, number
from ple.games.flappybird import FlappyBird
from ple import PLE
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt

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
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0



def observe_policy(agent, total_nb_episodes, numberOfObservations = 10, heatmap=True):
    reward_values = agent.reward_values()
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    env.init()

    nb_episodes = total_nb_episodes//numberOfObservations
    score = 0
    scores = []
    last_percent = 0
    now = datetime.now().strftime("%d-%m-%Y_%H %M %S")
    dirname = os.path.dirname(__file__)
    agentFolder = os.path.join(dirname, f"results/{now}")
    if not os.path.exists(agentFolder):
        os.makedirs(agentFolder)

    if not os.path.exists(agentFolder + "/plots"):
        os.makedirs(agentFolder + "/plots")

    for i in range(numberOfObservations):
        env.game.adjustRewards(reward_values)
        
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
            if score > 100:
                env.reset_game()
                nb_episodes -= 1
                score = 0
                
            # reset the environment if the game is over
            percent = (agent.episodes_observed / total_nb_episodes) * 100
            if (percent % 1 == 0 and percent != last_percent):
                last_percent = percent
                print("{}%".format(percent))
            if env.game_over():
                # print("score for this episode: %d" % score)
                env.reset_game()
                nb_episodes -= 1
                score = 0
        score = 0
        nb_episodes = 10
        tmp_scores= []
        env.game.adjustRewards({"positive": 1.0, "tick": 0.0, "loss": 0.0})
        while nb_episodes > 0 :
            action = agent.policy(env.game.getGameState())
            reward = env.act(env.getActionSet()[action])
            score += reward
            
            # if(score > 50):
            #     env.reset_game()
            #     tmp_scores.append(score)
            #     score = 0
            #     nb_episodes-=1
            
            if env.game_over():
                # print("score for this episode: %d" % score)
                env.reset_game()
                tmp_scores.append(score)
                score = 0
                nb_episodes-=1
            
        scores.append(tmp_scores)
        
        nb_episodes = total_nb_episodes//numberOfObservations
    if heatmap:
        agent.plot("pi")
        agent.fig.savefig(agentFolder + f"/plots/heatmap")
    # agent.lplot(scores, 10)
    # agent.fig.savefig(agentFolder + f"/plots/lineplot")
    df = pd.DataFrame({
        "mean score over 10 episodes" : [mean(score) for score in scores],
        # "max score over 10 episodes"  : [max(score) for score in scores]
    }, index = [(i+1) * (total_nb_episodes // numberOfObservations )for i in range(numberOfObservations)])
    ax = df.plot.line()
    ax.set_xlabel("Number of episodes trained")
    fig = ax.get_figure()
    fig.savefig(agentFolder + f"/plots/lineplot")
    
    print("saving pickle")
    with open (agentFolder + "/" + agent.__class__.__name__, 'wb') as pickle_file:
        pickle.dump(agent, pickle_file)

def observe_policy_task3(agent, total_nb_episodes, numberOfObservations = 10):
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
    fig = plt.figure()
    plt.plot(agent.loss, label="Log2 Loss")
    plt.legend()
    plt.xlabel("Number of episodes")
    fig.savefig(agentFolder + f"/plots/loss")
    
    print("saving pickle")
    with open (agentFolder + "/" + agent.__class__.__name__, 'wb') as pickle_file:
        pickle.dump(agent, pickle_file)

def observe_policy_50_score(agent, total_nb_episodes, numberOfObservations = 10, heatmap=True):
    reward_values = agent.reward_values()
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    env.init()

    nb_episodes = total_nb_episodes//numberOfObservations
    score = 0
    scores = []
    last_percent = 0
    now = datetime.now().strftime("%d-%m-%Y_%H %M %S")
    dirname = os.path.dirname(__file__)
    agentFolder = os.path.join(dirname, f"results/{now}")
    if not os.path.exists(agentFolder):
        os.makedirs(agentFolder)

    if not os.path.exists(agentFolder + "/plots"):
        os.makedirs(agentFolder + "/plots")

    for i in range(numberOfObservations):
        env.game.adjustRewards(reward_values)
        
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
            if score > 100:
                env.reset_game()
                nb_episodes -= 1
                score = 0
                
            # reset the environment if the game is over
            percent = (agent.episodes_observed / total_nb_episodes) * 100
            if (percent % 1 == 0 and percent != last_percent):
                last_percent = percent
                print("{}%".format(percent))
                
            if env.game_over():
                # print("score for this episode: %d" % score)
                env.reset_game()
                nb_episodes -= 1
                score = 0
                
        score = 0
        nb_episodes = 20
        tmp_scores= []
        env.game.adjustRewards({"positive": 1.0, "tick": 0.0, "loss": 0.0})
        while nb_episodes > 0 :
            action = agent.policy(env.game.getGameState())
            reward = env.act(env.getActionSet()[action])
            score += reward
            
            if(score > 60):
                env.reset_game()
                tmp_scores.append(score)
                score = 0
                nb_episodes-=1
            
            elif env.game_over():
                # print("score for this episode: %d" % score)
                env.reset_game()
                tmp_scores.append(score)
                score = 0
                nb_episodes-=1
            
        consistantFifty = False
        count = 0
        for sc in tmp_scores:
            if sc > 50: count+=1
        
        if count/len(tmp_scores) >= 0.9:
            print("Agent is trained and hit constant 50 score")
            print(f"Agent trained for {agent.frames_observed} frames" )
            print(f"Agent trained for {agent.episodes_observed} episodes" )
            
            print("saving pickle")
            with open (agentFolder + "/" + agent.__class__.__name__, 'wb') as pickle_file:
                pickle.dump(agent, pickle_file)
            return
            
        scores.append(tmp_scores)
        
        nb_episodes = total_nb_episodes//numberOfObservations
    if heatmap:
        agent.plot("pi")
        agent.fig.savefig(agentFolder + f"/plots/heatmap")
    # agent.lplot(scores, 10)
    # agent.fig.savefig(agentFolder + f"/plots/lineplot")
    df = pd.DataFrame({
        "mean score over 10 episodes" : [mean(score) for score in scores],
        # "max score over 10 episodes"  : [max(score) for score in scores]
    }, index = [(i+1) * (total_nb_episodes // numberOfObservations )for i in range(numberOfObservations)])
    ax = df.plot.line()
    ax.set_xlabel("Number of episodes trained")
    fig = ax.get_figure()
    fig.savefig(agentFolder + f"/plots/lineplot")
    
    print("saving pickle")
    with open (agentFolder + "/" + agent.__class__.__name__, 'wb') as pickle_file:
        pickle.dump(agent, pickle_file)

        
if __name__ == "__main__":

    # agent = TaskThreeAgent()
    # observe_policy_task3(agent, 50, 1)
    
    agent = Task4Agent2()
    observe_policy(agent, 300, 10, heatmap=False)
    # observe_policy_50_score(agent, 300, 20, heatmap=False)
    
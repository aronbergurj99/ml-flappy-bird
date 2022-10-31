import random
import math
import matplotlib.pyplot as plt
from numpy import block
import seaborn as sns
import pandas as pd
from sklearn.neural_network import MLPRegressor

class FlappyAgent:
    def __init__(self):
        self.actions = ("q_flap", "q_noop")

    def reward_values(self):
        """ returns the reward values used for training

            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        return

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        return random.randint(0, 1)

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        return random.randint(0, 1)


def split_to_interval(min, max, curr, intervals):
    if curr >= max: return intervals
    if curr <= 0: return 1
    val = (curr - min)/(max-min)
    return math.ceil(val * (intervals -1))

class BaseAgent(FlappyAgent):
    """
    Base Agent to not destroy the provided agent.
    """
    def __init__(self):
        super().__init__()
        self.interval = 15
        self.fig = None
        self.frames_observed = 0
        self.episodes_observed = 0


    def state_to_internal_state(self, state):
        return (
            split_to_interval(0, 512, state["next_pipe_top_y"], self.interval),
            split_to_interval(0, 512, state["player_y"], self.interval),
            max(-8, min(state["player_vel"], 10)), #clamp the value
            split_to_interval(0, 288, state["next_pipe_dist_to_player"], self.interval),
        )

    def q(s,a):
        pass

    def reward_values(self):
        return super().reward_values()

    def observe(self, s1, a, r, s2, end):
        return super().observe(s1, a, r, s2, end)

    def training_policy(self, state):
        state = self.state_to_internal_state(state)
        return super().training_policy(state)

    def policy(self, state):
        return super().policy(state)
        
    def lplot(self, scores, split):
            if self.fig == None:
                self.fig = plt.figure()
            else:
                plt.figure(self.fig.number)
            plt.clf()
            xData = []
            yData = []
            mData = []
            mean_score = 0
            max_score = 0
            counter = 0
            for i in range(len(scores)):
                if (max_score < scores[i]):
                    max_score = scores[i]
                mean_score += scores[i]
                counter += 1
                if (counter == split):
                    xData.append(i)
                    yData.append(mean_score / split)
                    mData.append(max_score)
                    max_score = 0
                    mean_score = 0
                    counter = 0
            df = {'Episodes':xData, 'Score':yData, 'Max Score':mData}
            sns.lineplot(data=df, x='Episodes', y='Score')
            sns.lineplot(data=df, x='Episodes', y='Max Score')
            plt.legend(labels=['Score', 'Max Score'])
    

class TaskTwoAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.q_values = {}
        self.initialize_q_values()
        self.epsilon = 0.1
        self.alpha = 0.1
        
    def initialize_q_values(self):
        """
        Creates the q table with all states actions equal to 0
        meaning that the agent knowns nothing about the env.
        """
        for player_y in range(1,self.interval+1):
            for next_pipe_top_y in range(1,self.interval+1):
                for next_pipe_dist_to_player in range(1,self.interval+1):
                    for player_vel in range(-8,11):
                        self.q_values[(next_pipe_top_y, player_y, player_vel, next_pipe_dist_to_player )] = [0, 0] #(flap, do nothing)

    def q(self, s,a):
        return self.q_values[self.state_to_internal_state(s)][a]

    def reward_values(self):
        return super().reward_values()

    def observe(self, s1, a, r, s2, end):
        self.frames_observed += 1
        if end : self.episodes_observed+=1
        self.q_values[self.state_to_internal_state(s1)][a]= self.q(s1,a) + self.alpha * (r + max([self.q(s2,0), self.q(s2,1)]) - self.q(s1,a))
            
    def training_policy(self, state):
        n = random.uniform(0, 1)
        if n < self.epsilon:
            return random.randint(0,1)
        else:
            actions = self.q_values[self.state_to_internal_state(state)]
            # print(actions)
            return actions.index(max(actions))

    def policy(self, state):
        actions = self.q_values[self.state_to_internal_state(state)]
        return actions.index(max(actions))
    
    def plot(self, what):
        data = [k + tuple(self.q_values[k]) for k in self.q_values.keys()]
        if self.fig == None:
            self.fig = plt.figure()
        else:
            plt.figure(self.fig.number)
            
        df = pd.DataFrame(data=data, columns=('next_pipe_top_y', 'player_y', 'player_vel', 'next_pipe_dist_to_player', 'q_flap', 'q_noop'))
        df['delta_y'] = df['player_y'] - df['next_pipe_top_y']
        df['v'] = df[['q_noop', 'q_flap']].max(axis=1)
        df['pi'] = (df[['q_noop', 'q_flap']].idxmax(axis=1) == 'q_flap')*1
        selected_data = df.groupby(['delta_y','next_pipe_dist_to_player'], as_index=False).mean()
        plt.clf()
        with sns.axes_style("white"):
            if what in ('q_flap', 'q_noop', 'v'):
                ax = sns.heatmap(selected_data.pivot('delta_y','next_pipe_dist_to_player',what), vmin=-5, vmax=5, cmap='coolwarm', annot=True, fmt='.2f')
            elif what == 'pi':
                ax = sns.heatmap(selected_data.pivot('delta_y','next_pipe_dist_to_player', 'pi'), vmin=0, vmax=1, cmap='coolwarm')
        ax.invert_xaxis()
        ax.set_title(what + ' after ' + str(self.frames_observed) + ' frames / ' + str(self.episodes_observed) + ' episodes')

def make_dummydata():
    return [random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)], [random.uniform(-5,5), random.uniform(-5,5)]

def normalize(
    value,
    min,
    max
):
    return 2*((value - min)/(max-min))-1

class TaskThreeAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.alpha = 0.001
        self.epsilon = 0.1
        
        self.regr = MLPRegressor(random_state=1, hidden_layer_sizes=(100,10), activation="logistic", alpha=self.alpha)
        dummy_data_x, dummy_data_y = make_dummydata()
        self.regr.partial_fit([dummy_data_x], [dummy_data_y])
        self.replayBufferSize = 1000
        self.batchSizse = 100
        self.replayBuffer = []
        self.loss = []
    
    def partial_fit(self, s1,a,r,s2,end):
        s1 = self.state_to_internal_state(s1)
        s2 = self.state_to_internal_state(s2)
        
        qa=0
        if not end:
            qa =  r + max(self.regr.predict([s2])[0])
        qb = self.regr.predict([s1])[0][1-a]
        
        actions = [qa if a==0 else qb, qb if a==0 else qa ]
        self.regr.partial_fit([s1], [actions])
        
    def observe(self, s1, a, r, s2, end):
        self.frames_observed += 1
        
        self.partial_fit(s1,a,r,s2,end)
        
        self.replayBuffer.append((s1,a,r,s2,end))
        if len(self.replayBuffer) > 1000:
            self.replayBuffer.pop(0)
        
        for i in range(self.batchSizse):
            self.partial_fit(*random.choice(self.replayBuffer))
        self.loss.append(math.log(self.regr.loss_, 2))
        
    def training_policy(self, state):
        state = self.state_to_internal_state(state)
        n = random.uniform(0, 1)
        if n < self.epsilon:
            return random.randint(0,1)
        else:
            actions = list(self.regr.predict([state])[0])
            return actions.index(max(actions))

    def policy(self, state):
        state = self.state_to_internal_state(state)
        # actions = self.q_values[self.state_to_internal_state(state)]
        # return actions.index(max(actions))
        actions = list(self.regr.predict([state])[0])
        return actions.index(max(actions))
        
    def state_to_internal_state(self, state):
        state = super().state_to_internal_state(state)
        return (
            normalize(state[0], 1, 15),
            normalize(state[1], 1, 15),
            normalize(state[2], -8, 10),
            normalize(state[3], 1, 15),
        )
        
class TaskFourAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.q_values = {}
        self.initialize_q_values()
        self.epsilon = 0.1
        self.alpha = 0.7
        self.gamma = 0.95
        self.realScore = 0
        
    def initialize_q_values(self):
        """
        Creates the q table with all states actions equal to 0
        meaning that the agent knowns nothing about the env.
        """
        for player_y in range(1,self.interval+1):
            for next_pipe_top_y in range(1,self.interval+1):
                for next_pipe_dist_to_player in range(1,self.interval+1):
                    for player_vel in range(-8,11):
                        self.q_values[(next_pipe_top_y, player_y, player_vel, next_pipe_dist_to_player )] = [0, 0] #(flap, do nothing)

    def q(self, s,a):
        return self.q_values[self.state_to_internal_state(s)][a]

    def reward_values(self):
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def observe(self, s1, a, r, s2, end):
        self.frames_observed += 1
        if end : self.episodes_observed+=1
        self.q_values[self.state_to_internal_state(s1)][a]= self.q(s1,a) + self.alpha * (r + self.gamma * max([self.q(s2,0), self.q(s2,1)]) - self.q(s1,a))
            
    def training_policy(self, state):

        n = random.uniform(0, 1)
        if n < self.epsilon:
            return random.randint(0,1)
        else:
            actions = self.q_values[self.state_to_internal_state(state)]
            # print(actions)
            return actions.index(max(actions))

    def policy(self, state):
        state =self.state_to_internal_state(state)
        actions = self.q_values[state]
        return actions.index(max(actions))
    
    def plot(self, what):
        data = [k + tuple(self.q_values[k]) for k in self.q_values.keys()]
        if self.fig == None:
            self.fig = plt.figure()
        else:
            plt.figure(self.fig.number)
            
        df = pd.DataFrame(data=data, columns=('next_pipe_top_y', 'player_y', 'player_vel', 'next_pipe_dist_to_player', 'q_flap', 'q_noop'))
        df['delta_y'] = df['player_y'] - df['next_pipe_top_y']
        df['v'] = df[['q_noop', 'q_flap']].max(axis=1)
        df['pi'] = (df[['q_noop', 'q_flap']].idxmax(axis=1) == 'q_flap')*1
        selected_data = df.groupby(['delta_y','next_pipe_dist_to_player'], as_index=False).mean()
        plt.clf()
        with sns.axes_style("white"):
            if what in ('q_flap', 'q_noop', 'v'):
                ax = sns.heatmap(selected_data.pivot('delta_y','next_pipe_dist_to_player',what), vmin=-5, vmax=5, cmap='coolwarm', annot=True, fmt='.2f')
            elif what == 'pi':
                ax = sns.heatmap(selected_data.pivot('delta_y','next_pipe_dist_to_player', 'pi'), vmin=0, vmax=1, cmap='coolwarm')
        ax.invert_xaxis()
        ax.set_title(what + ' after ' + str(self.frames_observed) + ' frames / ' + str(self.episodes_observed) + ' episodes')

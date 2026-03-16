import numpy as np

class POMDP:
    def __init__(self):
        self.states = [0, 1]  # tiger in door 0 or 1
        self.actions = ['listen', 'open door0', 'open door1']   # 0= open door 0, 1= open door 1
        self.observations = [0, 1]
        self.listen_cost = -20
        self.tiger_cost = -1600
        self.reward = 1000
        self.obs_prob = {
            0: {0: 0.75, 1: 0.25},  # if tiger is behind door 0, 75% chance of hearing it
            1: {0: 0.25, 1: 0.75}   # if tiger is behind door 1, 75% chance of hearing it
        }

    def reward_function(self, state, action):
        if action == 'listen':
            return self.listen_cost
        elif action == 'open door0':
            return self.tiger_cost if state == 0 else self.reward
        else:
            return self.tiger_cost if state == 1 else self.reward

    def obs_function(self, state, action, observation):
        if action == 'listen':
            return 0
        return self.obs_prob[state][observation]

    def belief_update(self, belief, action, observation):
        if action == 'listen':
            # Listening does not change the belief
            return belief
        new_belief = np.zeros(2)
        for s in self.states:
            new_belief[s] = belief[s] * self.obs_function(s, action, observation)
        if new_belief.sum() > 0:
            new_belief /= new_belief.sum()
        else:
            new_belief = np.array([0.5, 0.5])
        return new_belief

    def expected_reward(self, sequence, intial_belief=np.array([0.5, 0.5])):
        belief = intial_belief
        total_reward = 0

        for action in sequence:
            if action == 'listen':
                total_reward += self.listen_cost
                obs_prob = [0, 0]
                for obs in self.observations:
                    for state in self.states:
                        obs_prob[obs] += self.obs_prob[state][obs] * belief[state]
                new_beliefs = []
                for obs in self.observations:
                    if obs_prob[obs] > 0:
                        new_belief = self.belief_update(belief, action, obs)
                        new_beliefs.append(obs_prob[obs] * new_belief)
                if new_beliefs:
                    belief = np.sum(new_beliefs, axis=0)
            else:
                reward = 0
                for state in self.states:
                    reward += belief[state] * self.reward_function(state, action)
                total_reward += reward
                break
        return total_reward
    


    


        
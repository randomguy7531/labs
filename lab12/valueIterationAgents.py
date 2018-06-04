# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        for i in range(iterations):
            next_values = util.Counter()
            for state in mdp.getStates():
                q, _ = self.computeActionFromSuppliedValues(state, self.values)
                next_values[state] = q
            self.values = next_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromSuppliedValues(self, state, action, values):
        q = 0
        for transition_state, likelyhood in self.mdp.getTransitionStatesAndProbs(state, action):
            transition_reward = self.mdp.getReward(state, action, transition_state)
            q += (self.discount * values[transition_state] + transition_reward) * likelyhood
        return q

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        return self.computeQValueFromSuppliedValues(state, action, self.values)

    def computeActionFromSuppliedValues(self, state, values):
        actions = self.mdp.getPossibleActions(state)
        if len(actions) > 0 :
            q_values = []
            for action in actions:
                q_values.append(self.computeQValueFromSuppliedValues(state, action, values))
            actions_with_q = list(zip(q_values, actions))
            max_q_and_action = max(actions_with_q, key=(lambda action_with_q: action_with_q[0]))
            return max_q_and_action
        # handle the no actions case
        default_q = [0]
        default_action = [None]
        return list(zip(default_q, default_action))[0]

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        return self.computeActionFromSuppliedValues(state, self.values)[1]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

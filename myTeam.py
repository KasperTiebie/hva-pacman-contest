# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from util import manhattanDistance
from game import Directions
from util import nearestPoint
import game
import distanceCalculator

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'AlphaBetaAgent', second = 'AlphaBetaAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class AlphaBetaAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    def alphabeta(state, depth, agent, alpha, beta):
      if state.isOver():
        return self.evaluationFunction(state)

      if agent == state.getNumAgents():  # If this is the last agent
        if depth == 2:  # If we're at the required depth
          return self.evaluationFunction(state)
        else:  # Go one level deeper
          return alphabeta(state, depth + 1, 0, alpha, beta)

      if state.getAgentPosition(agent) is not None: # Check if agent is observable
        if len(state.getLegalActions(agent)) == 0:  # If the node doesn't have any successors (a terminal node)
          return self.evaluationFunction(state)

      if agent in self.getTeam(state):  # If agent is max
        value = float('-inf')
        for action in state.getLegalActions(agent):
          successor = state.generateSuccessor(agent, action)
          value = max([value, alphabeta(successor, depth, agent + 1, alpha, beta)])
          if value > beta:
            return value
          alpha = max([alpha, value])
        return value
      else:  # If agent is min
        value = float('inf')
        if state.getAgentPosition(agent) is not None:
          for action in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent, action)
            value = min([value, alphabeta(successor, depth, agent + 1, alpha, beta)])

            if value < alpha:  # If the value is lower than MAX's best option on path to root, return the value
              return value
            beta = min([beta, value])
        else:
          value = min([value, alphabeta(state, depth, agent + 1, alpha, beta)])
          if value < alpha:  # If the value is lower than MAX's best option on path to root, return the value
            return value
          beta = min([beta, value])

        return value

    best = None
    alpha = float('-inf')  # -inf
    beta = float('inf')  # +inf

    for action in gameState.getLegalActions(self.index):
      successor = gameState.generateSuccessor(self.index, action)
      if(self.index != gameState.getNumAgents()):
        value = alphabeta(successor, 1, self.index+1, alpha, beta)
      else:
        value = alphabeta(successor, 1, 0, alpha, beta)


      if value > alpha:
        alpha = value
        best = action

    return best

  def evaluationFunction(self, currentGameState):
    """
      The evaluation function
      minimumFoodDistanceSum: The sum of the distance to the nearest food pellet of each agent
      foodLeft: The amount of food left
      ghostNear: If a ghost is within a distance of 5
    """
    foodLeft = len(self.getFood(currentGameState).asList())

    if currentGameState.isOver():
      return 1000000

    if foodLeft > 2:
      weights = {'minimumFoodDistance': -0.1, 'foodLeft': -5.0, 'ghostNear': -10.0}
      features = util.Counter()

      features['foodLeft'] = foodLeft
      features['ghostNear'] = 0

      for agent in range(currentGameState.getNumAgents()-1):
        if agent in self.getTeam(currentGameState):
          features['minimumFoodDistance'] += min(self.getMazeDistance(currentGameState.getAgentPosition(agent), x) for x in self.getFood(currentGameState).asList())

          for enemyAgent in self.getOpponents(currentGameState):
            agentPosition = currentGameState.getAgentPosition(enemyAgent)
            if agentPosition is not None:
              features['ghostNear'] = 1
        #else:
          #return -10

      return features * weights
    else:
      weights = {'distanceToHome': -1.0}
      features = util.Counter()

      for agent in range(currentGameState.getNumAgents()-1):
        if agent in self.getTeam(currentGameState):
          features['distanceToHome'] += self.getMazeDistance(currentGameState.getAgentPosition(agent), currentGameState.getInitialAgentPosition(agent))

      return features * weights + 1000


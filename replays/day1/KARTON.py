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

CONTACT = "kasper@kaspertiebie.com"

from captureAgents import CaptureAgent
import random, time, util, math
from util import manhattanDistance
from game import Directions
from util import nearestPoint
import game
import distanceCalculator


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='offensiveAgent', second='defensiveAgent'):
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

class offensiveAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    self.isRed = gameState.isOnRedTeam(self.index)
    self.initialFood = self.getFood(gameState)

    self.layoutWidth = max(gameState.getWalls().asList(),
                           key=lambda x: x[0])[0]  # Returns the wall with the highest X value
    self.layoutHeight = max(gameState.getWalls().asList(),
                            key=lambda x: x[1])[1]  # Returns the wall with the highest Y value

    if self.isRed:
      self.middleX = math.floor(self.layoutWidth / 2)  # Returns the X value of the diving line
      self.middleY = math.floor(self.layoutHeight / 2)  # Returns the Y value of the imaginary diving Y line
      self.patrolAreaX = (self.middleX - math.floor(self.layoutWidth * 0.35), self.middleX)
      self.patrolAreaY = (self.middleY - math.floor(self.layoutHeight * 0.35), self.middleY + math.ceil(self.layoutHeight * 0.35))
    else:
      self.middleX = math.ceil(self.layoutWidth / 2)  # Returns the X value of the diving line
      self.middleY = math.ceil(self.layoutHeight / 2)  # Returns the Y value of the imaginary diving Y line
      self.patrolAreaX = (self.middleX, self.middleX + math.ceil(self.layoutWidth * 0.35))
      self.patrolAreaY = (self.middleY - math.floor(self.layoutHeight * 0.35), self.middleY + math.ceil(self.layoutHeight * 0.35))

    CaptureAgent.registerInitialState(self, gameState)

    # Marks the middle of the board
    self.debugDraw(cells=[(self.middleX, self.middleY)], color=[1, 0, 0])
    self.debugDraw(cells=[(self.patrolAreaX[0], self.patrolAreaY[0]),
                          (self.patrolAreaX[0], self.patrolAreaY[1]),
                          (self.patrolAreaX[1], self.patrolAreaY[0]),
                          (self.patrolAreaX[1], self.patrolAreaY[1])], color=[0, 1, 0])

  def chooseAction(self, gameState):
    """
    Returns the minimax action using self.depth and self.evaluationFunction
    """

    def alphabeta(state, depth, agent, alpha, beta, desiredDepth, desiredAgent):
      if state.isOver():
        return self.evaluationFunction(state)

      if agent is desiredAgent and depth is desiredDepth:
        return self.evaluationFunction(state)

      if agent is state.getNumAgents():  # If this is the last agent
        return alphabeta(state, depth + 1, 0, alpha, beta, desiredDepth, desiredAgent)

      if state.getAgentPosition(agent) is not None:  # Check if agent is obserable
        if len(state.getLegalActions(agent)) == 0:  # If the node doesn't have any successors (a terminal node)
          return self.evaluationFunction(state)

      if agent in self.getTeam(state):  # If agent is max
        value = float('-inf')
        for action in state.getLegalActions(agent):
          successor = state.generateSuccessor(agent, action)
          value = max([value, alphabeta(successor, depth, agent + 1, alpha, beta, desiredDepth, desiredAgent)])
          if value > beta:
            return value
          alpha = max([alpha, value])
        return value
      else:  # If agent is min
        value = float('inf')
        if state.getAgentPosition(agent) is not None:
          for action in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent, action)
            value = min([value, alphabeta(successor, depth, agent + 1, alpha, beta, desiredDepth, desiredAgent)])

            if value < alpha:  # If the value is lower than MAX's best option on path to root, return the value
              return value
            beta = min([beta, value])
        else:
          value = min([value, alphabeta(state, depth, agent + 1, alpha, beta, desiredDepth, desiredAgent)])
          if value < alpha:  # If the value is lower than MAX's best option on path to root, return the value
            return value

        return value

    bestActions = []
    alpha = float('-inf')  # -inf
    beta = float('inf')  # +inf

    #start = time.time()
    for action in gameState.getLegalActions(self.index):
      successor = gameState.generateSuccessor(self.index, action)
      if self.index is not gameState.getNumAgents():
        value = alphabeta(successor, 1, self.index + 1, alpha, beta, 2, self.index)
      else:
        value = alphabeta(successor, 1, 0, alpha, beta, 2, self.index)

      if value > alpha:
        alpha = value
        bestActions = []
        bestActions.append(action)
      elif value == alpha:
        bestActions.append(action)

    #print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    return random.choice(bestActions)

  def isOnOwnSide(self, gameState):
    if self.isRed:
      return gameState.getAgentPosition(self.index)[0] < self.middleX
    else:
      return gameState.getAgentPosition(self.index)[0] > self.middleX

  def evaluationFunction(self, currentGameState):
    """
          The evaluation function
          minimumFoodDistanceSum: The sum of the distance to the nearest food pellet of each agent
          foodLeft: The amount of food left
          ghostNear: If a ghost is within a distance of 2
        """
    foodLeft = len(self.getFood(currentGameState).asList())
    foodCarrying = currentGameState.data.agentStates[self.index].numCarrying
    foodReturned = currentGameState.data.agentStates[self.index].numReturned
    isPacman = currentGameState.data.agentStates[self.index].isPacman
    agentPosition = currentGameState.getAgentPosition(self.index)

    # If the game is over at this state
    if currentGameState.isOver():
      if foodLeft <= 2:  # If our team is winning
        return 1000000  # Make sure that happens by returning a high value
      elif len(self.getFoodYouAreDefending(currentGameState).asList()) <= 2:  # If the opposing team is winning
        return -1000000  # Make sure that doesn't happen by returning a low value

    # Collect more food if the foodLeft is > 2 and the agent isn't holding more than 2 food
    weights = weights = {'distanceToHome': -0.3*foodCarrying,
                         'minimumFoodDistance': -1.0/(1+foodCarrying),
                         'foodLeft': -10.0,
                         'ghostDistance': 3.0,
                         'foodCarrying': 22+(foodReturned * 1),
                         'foodReturned': 22.0 + (foodReturned * 2)}
    features = util.Counter()
    features['foodReturned'] = foodReturned
    features['foodCarrying'] = foodCarrying
    features['foodLeft'] = foodLeft
    features['ghostDistance'] = 15

    features['minimumFoodDistance'] = min(
      self.getMazeDistance(agentPosition, x) for x in self.getFood(currentGameState).asList())

    if isPacman:
      for enemyAgent in self.getOpponents(currentGameState):
        enemyAgentPosition = currentGameState.getAgentPosition(enemyAgent)
        if enemyAgentPosition is not None:
          enemyDistance = self.getMazeDistance(enemyAgentPosition, agentPosition)
          if enemyDistance < features['ghostDistance']:
            features['ghostDistance'] = enemyDistance

    features['distanceToHome'] = self.getMazeDistance(currentGameState.getAgentPosition(self.index),
                                                      currentGameState.getInitialAgentPosition(self.index))

    return features * weights

class defensiveAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    self.isRed = gameState.isOnRedTeam(self.index)

    self.layoutWidth = max(gameState.getWalls().asList(), key=lambda x: x[0])[0]  # Returns the wall with the highest X value
    self.layoutHeight = max(gameState.getWalls().asList(), key=lambda x: x[1])[1]  # Returns the wall with the highest Y value

    if self.isRed:
      self.middleX = math.floor(self.layoutWidth / 2)  # Returns the X value of the diving line
      self.middleY = math.floor(self.layoutHeight / 2)  # Returns the Y value of the imaginary diving Y line
      self.patrolAreaX = (self.middleX - math.floor(self.layoutWidth * 0.35), self.middleX)
      self.patrolAreaY = (
      self.middleY - math.floor(self.layoutHeight * 0.35), self.middleY + math.ceil(self.layoutHeight * 0.35))
    else:
      self.middleX = math.ceil(self.layoutWidth / 2)  # Returns the X value of the diving line
      self.middleY = math.ceil(self.layoutHeight / 2)  # Returns the Y value of the imaginary diving Y line
      self.patrolAreaX = (self.middleX, self.middleX + math.ceil(self.layoutWidth * 0.35))
      self.patrolAreaY = (
      self.middleY - math.floor(self.layoutHeight * 0.35), self.middleY + math.ceil(self.layoutHeight * 0.35))

    CaptureAgent.registerInitialState(self, gameState)

    # Marks the middle of the board
    self.debugDraw(cells=[(self.middleX, self.middleY)], color=[1, 0, 0])
    self.debugDraw(cells=[(self.patrolAreaX[0], self.patrolAreaY[0]),
                          (self.patrolAreaX[0], self.patrolAreaY[1]),
                          (self.patrolAreaX[1], self.patrolAreaY[0]),
                          (self.patrolAreaX[1], self.patrolAreaY[1])], color=[0, 1, 0])

  def chooseAction(self, gameState):
    """
    Returns the minimax action using self.depth and self.evaluationFunction
    """

    def alphabeta(state, depth, agent, alpha, beta, desiredDepth, desiredAgent):
      if state.isOver():
        return self.evaluationFunction(state)

      if agent is desiredAgent and depth is desiredDepth:
        return self.evaluationFunction(state)

      if agent is state.getNumAgents():  # If this is the last agent
        return alphabeta(state, depth + 1, 0, alpha, beta, desiredDepth, desiredAgent)

      if state.getAgentPosition(agent) is not None:  # Check if agent is observable
        if len(state.getLegalActions(agent)) == 0:  # If the node doesn't have any successors (a terminal node)
          return self.evaluationFunction(state)

      if agent in self.getTeam(state):  # If agent is max
        value = float('-inf')
        for action in state.getLegalActions(agent):
          successor = state.generateSuccessor(agent, action)
          value = max([value, alphabeta(successor, depth, agent + 1, alpha, beta, desiredDepth, desiredAgent)])
          if value > beta:
            return value
          alpha = max([alpha, value])
        return value
      else:  # If agent is min
        value = float('inf')
        if state.getAgentPosition(agent) is not None:
          for action in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent, action)
            value = min([value, alphabeta(successor, depth, agent + 1, alpha, beta, desiredDepth, desiredAgent)])

            if value < alpha:  # If the value is lower than MAX's best option on path to root, return the value
              return value
            beta = min([beta, value])
        else:
          value = min([value, alphabeta(state, depth, agent + 1, alpha, beta, desiredDepth, desiredAgent)])
          if value < alpha:  # If the value is lower than MAX's best option on path to root, return the value
            return value

        return value

    bestActions = []
    alpha = float('-inf')  # -inf
    beta = float('inf')  # +inf

    #start = time.time()
    for action in gameState.getLegalActions(self.index):
      successor = gameState.generateSuccessor(self.index, action)
      if self.index is not gameState.getNumAgents():
        value = alphabeta(successor, 1, self.index + 1, alpha, beta, 2, self.index)
      else:
        value = alphabeta(successor, 1, 0, alpha, beta, 2, self.index)

      if value > alpha:
        alpha = value
        bestActions = []
        bestActions.append(action)
      elif value == alpha:
        bestActions.append(action)

    #print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    return random.choice(bestActions)

  def isOnOwnSide(self, gameState):
    if self.isRed:
      return gameState.getAgentPosition(self.index)[0] < self.middleX
    else:
      return gameState.getAgentPosition(self.index)[0] > self.middleX

  def evaluationFunction(self, currentGameState):
    """
          The evaluation function
          minimumFoodDistanceSum: The sum of the distance to the nearest food pellet of each agent
          foodLeft: The amount of food left
          ghostNear: If a ghost is within a distance of 2
        """
    foodDefendingLeft = len(self.getFoodYouAreDefending(currentGameState).asList())
    isPacman = currentGameState.data.agentStates[self.index].isPacman
    agentPosition = currentGameState.getAgentPosition(self.index)

    # If the game is over at this state
    if currentGameState.isOver():
      if foodDefendingLeft <= 2:  # If the opposing team is winning
        return -1000000  # Make sure that doesn't happen by returning a low
      elif len(self.getFood(currentGameState).asList()) <= 2:  # If our team is winning
        return 1000000  # Make sure that happens by returning a high value

    weights = {'foodDefendingLeft': 50,
               'distanceToEnemy': -5.0,
               'isPacman': -10000,
               'distanceToPatrolArea': -1.0}
    features = util.Counter()
    features['foodDefendingLeft'] = foodDefendingLeft
    features['distanceToPatrolArea'] = float('inf')
    features['distanceToEnemy'] = 20

    if isPacman: features['isPacman'] = 1

    # If the agent is outside of the patrol area, set 'distanceToPatrolArea' equal to the minimum distance to the patrol area
    if agentPosition[0] < self.patrolAreaX[0] or agentPosition[0] > self.patrolAreaX[1] or agentPosition[1] < self.patrolAreaY[0] or agentPosition[1] > self.patrolAreaY[1]:
      for y in range(self.patrolAreaY[0], self.patrolAreaY[1]):
        if not currentGameState.hasWall(self.patrolAreaX[0], y):
          distance = self.getMazeDistance(agentPosition, (self.patrolAreaX[0], y))
          if distance < features['distanceToPatrolArea']: features['distanceToPatrolArea'] = distance
        if not currentGameState.hasWall(self.patrolAreaX[1], y):
          distance = self.getMazeDistance(agentPosition, (self.patrolAreaX[1], y))
          if distance < features['distanceToPatrolArea']: features['distanceToPatrolArea'] = distance

      for x in range(self.patrolAreaX[0], self.patrolAreaX[1]):
        if not currentGameState.hasWall(x, self.patrolAreaY[0]):
          distance = self.getMazeDistance(agentPosition, (x, self.patrolAreaY[0]))
          if distance < features['distanceToPatrolArea']: features['distanceToPatrolArea'] = distance
        if not currentGameState.hasWall(x, self.patrolAreaY[1]):
          distance = self.getMazeDistance(agentPosition, (x, self.patrolAreaY[1]))
          if distance < features['distanceToPatrolArea']: features['distanceToPatrolArea'] = distance
    else:
      features['distanceToPatrolArea'] = 0

    # Loops through all enemies and sets 'distanceToGhost' equal to the lowest distance to a ghost
    for enemyAgent in self.getOpponents(currentGameState):
      enemyAgentPosition = currentGameState.getAgentPosition(enemyAgent)
      if enemyAgentPosition is not None:
        distanceToEnemyAgent = self.getMazeDistance(enemyAgentPosition, agentPosition)
        if distanceToEnemyAgent < features['distanceToEnemy']: features['distanceToEnemy'] = distanceToEnemyAgent

    return features * weights

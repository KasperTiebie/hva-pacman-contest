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
desiredDepth = 2
maxFood = 0
hasBeenHome = True
# foodCountVerified = False


class offensiveAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    self.isRed = gameState.isOnRedTeam(self.index)

    self.layoutWidth = max(gameState.getWalls().asList(), key=lambda x: x[0])[
      0]  # Returns the wall with the highest X value
    self.layoutHeight = max(gameState.getWalls().asList(), key=lambda x: x[1])[
      1]  # Returns the wall with the highest Y value
    self.middleX = round(self.layoutWidth / 2)  # Returns the X value of the diving line
    self.middleY = round(self.layoutHeight / 2)  # Returns the Y value of the imaginary diving Y line

    if self.isRed:
      self.patrolAreaX = (self.middleX - math.floor(self.layoutWidth * 0.35), self.middleX)
      self.patrolAreaY = (self.middleY - math.floor(self.layoutHeight * 0.35), self.middleY + math.ceil(self.layoutHeight * 0.35))
    else:
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

    def alphabeta(state, depth, agent, alpha, beta):

      if state.isOver():
        return self.evaluationFunction(state)

      if agent == state.getNumAgents():  # If this is the last agent
        global desiredDepth
        if depth == desiredDepth:  # If we're at the required depth
          return self.evaluationFunction(state)
        else:  # Go one level deeper
          return alphabeta(state, depth + 1, 0, alpha, beta)

      if state.getAgentPosition(agent) is not None:  # Check if agent is obserable
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

        return value

    best = None
    alpha = float('-inf')  # -inf
    beta = float('inf')  # +inf

    for action in gameState.getLegalActions(self.index):
      successor = gameState.generateSuccessor(self.index, action)
      successor.getAgentPosition(self.index)
      if self.index is not gameState.getNumAgents():
        value = alphabeta(successor, 1, self.index + 1, alpha, beta)
      else:
        value = alphabeta(successor, 1, 0, alpha, beta)

      if value > alpha:
        alpha = value
        best = action

    return best

  def isOnOwnSide(self, gameState):
    if self.isRed:
      return gameState.getAgentPosition(self.index)[0] < self.middleX - 1
    else:
      return gameState.getAgentPosition(self.index)[0] > self.middleX + 1

  def evaluationFunction(self, currentGameState):
    """
          The evaluation function
          minimumFoodDistanceSum: The sum of the distance to the nearest food pellet of each agent
          foodLeft: The amount of food left
          ghostNear: If a ghost is within a distance of 2
        """
    foodLeft = len(self.getFood(currentGameState).asList())

    global maxFood
    global hasBeenHome

    if maxFood == 0:
      maxFood = foodLeft

    # If the game is over at this state
    if currentGameState.isOver():
      if foodLeft <= 2:  # If our team is winning
        return 1000000  # Make sure that happens by returning a high value
      elif len(self.getFoodYouAreDefending(currentGameState).asList()) <= 2:  # If the opposing team is winning
        return -1000000  # Make sure that doesn't happen by returning a low value

    # Collect more food if the foodLeft is > 2 and the agent isn't holding more than 2 food

    if maxFood - foodLeft < 0: maxFood = foodLeft

    if foodLeft > 2 and maxFood - foodLeft < 5 and hasBeenHome:
      weights = {'minimumFoodDistance': -0.1, 'foodLeft': -5.0, 'ghostNear': -100.0}

      features = util.Counter()

      features['foodLeft'] = foodLeft
      features['ghostNear'] = 0

      agentPosition = currentGameState.getAgentPosition(self.index)
      features['minimumFoodDistance'] += min(
        self.getMazeDistance(agentPosition, x) for x in self.getFood(currentGameState).asList())

      for enemyAgent in self.getOpponents(currentGameState):
        enemyAgentPosition = currentGameState.getAgentPosition(enemyAgent)
        if enemyAgentPosition is not None:
          if self.getMazeDistance(enemyAgentPosition, agentPosition) <= 4:
            if self.isOnOwnSide(currentGameState):
              weights = {'minimumFoodDistance': -0.1, 'foodLeft': -5.0,
                         'ghostNear': -100.0}  # look for food, but definitely hunt for any enemies nearby.(when safely crossed the border ofc....)
            else:
              weights = {'minimumFoodDistance': -0.0, 'foodLeft': -0.0,
                         'ghostNear': -100.0}  # prioritize your life over food, so you don't get caught in dead ends.
            features['ghostNear'] = 1

      return features * weights

    # Go home to return the food
    else:
      weights = {'distanceToHome': -1.0, 'distanceToGhost': -50.0}
      features = util.Counter()

      features['distanceToHome'] = self.getMazeDistance(currentGameState.getAgentPosition(self.index),
                                                        currentGameState.getInitialAgentPosition(self.index))

      # Loops through all enemies and sets 'distanceToGhost' equal to the lowest distance to a ghost
      agentPosition = currentGameState.getAgentPosition(self.index)
      for enemyAgent in self.getOpponents(currentGameState):
        enemyAgentPosition = currentGameState.getAgentPosition(enemyAgent)
        if enemyAgentPosition is not None:
          distanceToEnemyAgent = self.getMazeDistance(enemyAgentPosition, agentPosition)
          if distanceToEnemyAgent < features['distanceToGhost']:
            features['distanceToGhost'] = distanceToEnemyAgent

      if self.isOnOwnSide(currentGameState):
        hasBeenHome = True
        maxFood = 0
      else:
        hasBeenHome = False

      return features * weights + 1000


class defensiveAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    self.isRed = gameState.isOnRedTeam(self.index)

    self.layoutWidth = max(gameState.getWalls().asList(), key=lambda x: x[0])[0]  # Returns the wall with the highest X value
    self.layoutHeight = max(gameState.getWalls().asList(), key=lambda x: x[1])[1]  # Returns the wall with the highest Y value
    self.middleX = round(self.layoutWidth / 2)  # Returns the X value of the diving line
    self.middleY = round(self.layoutHeight / 2)  # Returns the Y value of the imaginary diving Y line

    if self.isRed:
      self.patrolAreaX = (self.middleX - math.floor(self.layoutWidth * 0.35), self.middleX)
      self.patrolAreaY = (self.middleY - math.floor(self.layoutHeight * 0.35), self.middleY + math.ceil(self.layoutHeight * 0.35))
    else:
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

    def alphabeta(state, depth, agent, alpha, beta):
      if state.isOver():
        return self.evaluationFunction(state)

      if agent == state.getNumAgents():  # If this is the last agent
        global desiredDepth
        if depth == desiredDepth:  # If we're at the required depth
          return self.evaluationFunction(state)
        else:  # Go one level deeper
          return alphabeta(state, depth + 1, 0, alpha, beta)

      if state.getAgentPosition(agent) is not None:  # Check if agent is observable
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

        return value

    bestActions = []
    alpha = float('-inf')  # -inf
    beta = float('inf')  # +inf

    for action in gameState.getLegalActions(self.index):
      successor = gameState.generateSuccessor(self.index, action)
      if self.index is not gameState.getNumAgents():
        value = alphabeta(successor, 1, self.index + 1, alpha, beta)
      else:
        value = alphabeta(successor, 1, 0, alpha, beta)

      if value > alpha:
        alpha = value
        bestActions = []
        bestActions.append(action)
      elif value == alpha:
        bestActions.append(action)

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

    # If the game is over at this state
    if currentGameState.isOver():
      if foodDefendingLeft <= 2:  # If the opposing team is winning
        return -1000000  # Make sure that doesn't happen by returning a low
      elif len(self.getFood(currentGameState).asList()) <= 2:  # If our team is winning
        return 1000000  # Make sure that happens by returning a high value

    weights = {'foodDefendingLeft': 50, 'distanceToEnemy': -5.0, 'isOnEnemySide': -10000,
               'distanceToPatrolAreaX': -1.0, 'distanceToPatrolAreaY': -1.0}
    features = util.Counter()

    features['foodDefendingLeft'] = foodDefendingLeft
    features['distanceToPatrolAreaX'] = float('inf')
    features['distanceToPatrolAreaY'] = float('inf')
    features['distanceToEnemy'] = 20

    agentPosition = currentGameState.getAgentPosition(self.index)

    # If the agent is outside of the X patrol area
    if agentPosition[0] < self.patrolAreaX[0] or agentPosition[0] > self.patrolAreaX[1]:
      for y in range(self.layoutHeight):
        if not currentGameState.hasWall(self.patrolAreaX[0], y):
          distance = self.getMazeDistance(agentPosition, (self.patrolAreaX[0], y))
          if distance < features['distanceToPatrolAreaX']: features['distanceToPatrolAreaX'] = distance
        if not currentGameState.hasWall(self.patrolAreaX[1], y):
          distance = self.getMazeDistance(agentPosition, (self.patrolAreaX[1], y))
          if distance < features['distanceToPatrolAreaX']: features['distanceToPatrolAreaX'] = distance
    else:
      features['distanceToPatrolAreaX'] = 0

    # If the agent is outside of the Y patrol area
    if agentPosition[1] < self.patrolAreaY[0] or agentPosition[1] > self.patrolAreaY[1]:
      for x in range(self.layoutWidth):
        if not currentGameState.hasWall(x, self.patrolAreaY[0]):
          distance = self.getMazeDistance(agentPosition, (x, self.patrolAreaY[0]))
          if distance < features['distanceToPatrolAreaY']: features['distanceToPatrolAreaY'] = distance
        if not currentGameState.hasWall(x, self.patrolAreaY[1]):
          distance = self.getMazeDistance(agentPosition, (x, self.patrolAreaY[1]))
          if distance < features['distanceToPatrolAreaY']: features['distanceToPatrolAreaY'] = distance
    else:
      features['distanceToPatrolAreaY'] = 0

    if not self.isOnOwnSide(currentGameState):
      features['isOnEnemySide'] = 1

    # Loops through all enemies and sets 'distanceToGhost' equal to the lowest distance to a ghost
    for enemyAgent in self.getOpponents(currentGameState):
      enemyAgentPosition = currentGameState.getAgentPosition(enemyAgent)
      if enemyAgentPosition is not None:
        distanceToEnemyAgent = self.getMazeDistance(enemyAgentPosition, agentPosition)
        if distanceToEnemyAgent < features['distanceToEnemy']: features['distanceToEnemy'] = distanceToEnemyAgent

    return features * weights

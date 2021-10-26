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
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'offensiveAgent', second = 'defensiveAgent'):
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

class Node:
  def __init__(self, state, action, cost, parent):
    self.state = state
    self.action = action
    self.cost = cost
    self.parent = parent

  def getActionPath(self):
    if self.parent is None:
      return []
    else:
      action_path = self.parent.getActionPath()
      action_path.append(self.action)
      return action_path

def aStarSearch(index, gameState, goalState):
  current_node = Node(gameState.getAgentPosition(index), None, 0, None)
  closed = set([])
  fringe = util.PriorityQueue()
  fringe.push(current_node, current_node.cost)
  print("NEW")
  while True:
    if fringe.isEmpty():
      return False

    node = fringe.pop()

    if node.state not in closed:
      closed.add(node.state)

      if node.state == goalState:
        return node.getActionPath()

      for action in gameState.getLegalActions(index):
        s = gameState.generateSuccessor(index, action).getAgentPosition(index)
        print("-------------------")
        print("Old state: " + str(node.state))
        print("Old state: " + str(gameState.getAgentPosition(index)))
        print("Action: " + str(action))
        print("New state: " + str(s))
        #joe = gameState.generateSuccessor(index, action)
        successor_node = Node(s, action, 1, node)
        if successor_node.state not in closed:
          fringe.push(successor_node, successor_node.cost + manhattanDistance(successor_node.state, goalState))

class defensiveAgent(CaptureAgent):
  # couple = []
  # coupleMade = False
  # for oponentIndex in getOpponents():
  #   if getAgentPosition(oponentIndex) in getFoodYouAreDefending():
  #     couple = [CaptureAgent.index, oponentIndex]
  #     cupleMade = True
  #     fightThisMotherFucker(couple)

  # def fightThisMotherFucker(couple):
  #   getShortestPath(getAgentPosition(couple[1])

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    nearest_food = min(gameState.getRedFood().asList(), key=lambda x: manhattanDistance(gameState.getAgentPosition(self.index), x))
    #for o in CaptureAgent.getOpponents(gameState):
    print(aStarSearch(self.index, gameState, nearest_food))

    actions = gameState.getLegalActions(self.index)
    return random.choice(actions)


class offensiveAgent(CaptureAgent):
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    return random.choice(actions)

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)


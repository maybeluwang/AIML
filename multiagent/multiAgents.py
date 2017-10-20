# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = [action for action in gameState.getLegalActions() if action != 'Stop']

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"


        return legalMoves[chosenIndex]


    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        if successorGameState.isWin():
          return 999999999
        if successorGameState.isLose():
          return -999999999
        ghostDistances, capsuleDistanceList = AllGhostsCapsulesDistance(successorGameState)
        weightFood, weightGhost, weightCapsule, weightHunter = 10.0, 5.0, 5.0, 0.0
        ghostScore, capsuleScore, hunterScore = 0.0, 0.0, 0.0

        foodScore = 1.0 / shortestFoodDistance(successorGameState)

        GhostStates = successorGameState.getGhostStates()

        GhostPoses = successorGameState.getGhostPositions()
        ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

        if len(capsuleDistanceList) != 0:
          capsuleScore = 1.0 / min(capsuleDistanceList)
          if min(capsuleDistanceList)*2 < min(ghostDistances):
            weightCapsule = 10.0

        for index, ScaredTimer in enumerate(ScaredTimes):
          if ScaredTimer > ghostDistances[index]*3:
              weightHunter = 20.0
              hunterScore += 1.0
          else:
            if ghostDistances[index] < 3:
              ghostScore = -100.0
            elif ghostDistances[index] == 4:
              ghostScore = -10.0
            elif ghostDistances[index] == 5 and len(capsuleDistanceList) != 0:
              if min(capsuleDistanceList) > 2:
                ghostScore = -5.0
    
        heuristic = successorGameState.getScore()+ weightFood*foodScore + weightGhost*ghostScore + weightCapsule*capsuleScore + weightHunter*hunterScore

        return heuristic   

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
            agentIndex=0 -> MAX
            agentIndex>=1 -> MIN

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
          Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        TotalNumAgent = gameState.getNumAgents()
        TotalDepth = self.depth*TotalNumAgent

        def MiniMax(gameState, depth, agentIndex):
          if depth == 0 or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)

          bestAction = None
          legalMoves = [action for action in gameState.getLegalActions(agentIndex) if action != 'Stop']
          successorGameStates = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
          if agentIndex == 0: # Pacman Max Player
            bestValue = float("-inf") 
            for index, state in enumerate(successorGameStates):
              (value, _) = MiniMax(state, depth-1, agentIndex+1)
              if value > bestValue:
                bestValue, bestAction = value, legalMoves[index]
            return (bestValue, bestAction)
          else:
            bestValue = float("inf")
            for index, state in enumerate(successorGameStates):
              if agentIndex == (TotalNumAgent-1):
                (value, _) = MiniMax(state, depth-1, 0)
              else:
                (value, _) = MiniMax(state, depth-1, agentIndex+1)
              if value < bestValue:
                bestValue, bestAction = value, legalMoves[index]
            return (bestValue, bestAction)

        (_, ReturnAction) = MiniMax(gameState, TotalDepth, 0)
        return ReturnAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        TotalNumAgent = gameState.getNumAgents()
        TotalDepth = self.depth*TotalNumAgent

        def alphabeta(gameState, depth, alpha, beta, agentIndex):
          if depth == 0 or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)

          legalMoves = [action for action in gameState.getLegalActions(agentIndex) if action != 'Stop']
          successorGameStates = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
          bestAction = None
          if agentIndex == 0: # Pacman Max Player
            bestValue = float("-inf")
            for index, state in enumerate(successorGameStates):
              (value, _) = alphabeta(state, depth-1, alpha, beta, agentIndex+1)
              if value > bestValue:
                bestValue, bestAction = value, legalMoves[index]
              if value > alpha:
                alpha = value
              if beta <= alpha:
                break
            return (bestValue, bestAction)
          else:
            bestValue = float("inf")
            for index, state in enumerate(successorGameStates):
              if agentIndex == (TotalNumAgent-1):
                (value, _) = alphabeta(state, depth-1, alpha, beta, 0)
              else:
                (value, _) = alphabeta(state, depth-1, alpha, beta, agentIndex+1)
              if value < bestValue:
                bestValue, bestAction = value, legalMoves[index]
              if value < beta:
                beta = value
              if beta <= alpha:
                break
            return (bestValue, bestAction)

        (_, action) = alphabeta(gameState, TotalDepth, float("-inf"), float("inf"), 0)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        TotalNumAgent = gameState.getNumAgents()
        TotalDepth = self.depth*TotalNumAgent

        def ExpectiMinimax(gameState, depth, agentIndex):
          if depth == 0 or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)

          bestAction = None
          legalMoves = [action for action in gameState.getLegalActions(agentIndex) if action != 'Stop']

          successorGameStates = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
          if agentIndex == 0: # Pacman Max Player
            bestValue = float("-inf") 
            for index, state in enumerate(successorGameStates):
              (value, _) = ExpectiMinimax(state, depth-1, agentIndex+1)
              if value > bestValue:
                bestValue, bestAction = value, legalMoves[index]
            return (bestValue, bestAction)
          else:
            bestValue = 0
            for state in successorGameStates:
              if agentIndex == (TotalNumAgent-1):
                (value, _) = ExpectiMinimax(state, depth-1, 0)
              else:
                (value, _) = ExpectiMinimax(state, depth-1, agentIndex+1)
              bestValue += float(value)/float(len(successorGameStates))
            return (bestValue, bestAction)

        (_, ReturnAction) = ExpectiMinimax(gameState, TotalDepth, 0)
        return ReturnAction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    if currentGameState.isWin():
      return 999999999
    if currentGameState.isLose():
      return -999999999

    ghostDistances, capsuleDistanceList = AllGhostsCapsulesDistance(currentGameState)
    weightFood, weightGhost, weightCapsule, weightHunter = 5.0, 5.0, 5.0, 0.0
    ghostScore, capsuleScore, hunterScore = 0.0, 0.0, 0.0

    foodScore = 1.0 / shortestFoodDistance(currentGameState)

    GhostStates = currentGameState.getGhostStates()

    GhostPoses = currentGameState.getGhostPositions()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

    if len(capsuleDistanceList) != 0:
      capsuleScore = 1.0 / min(capsuleDistanceList)

    for index, ScaredTimer in enumerate(ScaredTimes):
      if ScaredTimer != 0:
          ScaredTimer < ghostDistances[index]
          weightHunter = 20.0
          hunterScore += 1.0
      else:
          ghostScore = -1.0/ghostDistances[index]

    
    "a new evaluation function."
    heuristic = currentGameState.getScore()+ weightFood*foodScore + weightGhost*ghostScore + weightCapsule*capsuleScore + weightHunter*hunterScore

    return heuristic    


def AllGhostsCapsulesDistance(GameState):
    from util import Queue
    frontier = Queue()

    position = GameState.getPacmanPosition()
    
    NotFound = GameState.getGhostPositions()+ GameState.getCapsules()

    frontier.push(position)
    explored = set()
    transitionTable = dict()
    while ( True ):
      if frontier.isEmpty():
        break
      x, y = frontier.pop()
      if (x, y) in NotFound:
        NotFound.remove((x,y))
      if len(NotFound) == 0:
        break
      explored.add((x, y))
      if not GameState.hasWall(x-1, y):
        if not ((x-1, y) in explored or transitionTable.has_key((x-1, y))):
          frontier.push((x-1, y))
          transitionTable[(x-1, y)] = (x, y)

      if not GameState.hasWall(x+1, y):
        if not ((x+1, y) in explored or transitionTable.has_key((x+1, y))):
          frontier.push((x+1, y))
          transitionTable[(x+1, y)] = (x, y)

      if not GameState.hasWall(x, y-1):
        if not ((x, y-1) in explored or transitionTable.has_key((x, y-1))):
          frontier.push((x, y-1))
          transitionTable[(x, y-1)] = (x, y)

      if not GameState.hasWall(x, y+1):
        if not ((x, y+1) in explored or transitionTable.has_key((x, y+1))):
          frontier.push((x, y+1))
          transitionTable[(x, y+1)] = (x, y)

    ghostDistanceList = []
    capsuleDistanceList = []
    GhostsPos = GameState.getGhostPositions()
    CapsulesPos = GameState.getCapsules()
    for ghostPos in GhostsPos:
      count = 0
      child = ghostPos    
      while ( True ):
        parent = transitionTable.get(child)
        if parent == None:
          break
        count += 1
        child = parent
      ghostDistanceList.append(count)

    for capsule in CapsulesPos:
      count = 0
      child = capsule    
      while ( True ):
        parent = transitionTable.get(child)
        if parent == None:
          break
        count += 1
        child = parent
      capsuleDistanceList.append(count)
    #print GameState
    #print ghostDistanceList
    #print capsuleDistanceList

    return ghostDistanceList, capsuleDistanceList

def AllGhostsDistance(GameState):
    from util import Queue
    frontier = Queue()

    position = GameState.getPacmanPosition()
    GhostsPos = GameState.getGhostPositions()
    GhostsPosNotFind =  GameState.getGhostPositions()
    frontier.push(position)
    explored = set()
    transitionTable = dict()
    while ( True ):
      if frontier.isEmpty():
        return float("inf")
      x, y = frontier.pop()
      if (x, y) in GhostsPosNotFind:
        GhostsPosNotFind.remove((x,y))
        if len(GhostsPosNotFind) == 0:
          break
      explored.add((x, y))
      if not GameState.hasWall(x-1, y):
        if not ((x-1, y) in explored or transitionTable.has_key((x-1, y))):
          frontier.push((x-1, y))
          transitionTable[(x-1, y)] = (x, y)

      if not GameState.hasWall(x+1, y):
        if not ((x+1, y) in explored or transitionTable.has_key((x+1, y))):
          frontier.push((x+1, y))
          transitionTable[(x+1, y)] = (x, y)

      if not GameState.hasWall(x, y-1):
        if not ((x, y-1) in explored or transitionTable.has_key((x, y-1))):
          frontier.push((x, y-1))
          transitionTable[(x, y-1)] = (x, y)

      if not GameState.hasWall(x, y+1):
        if not ((x, y+1) in explored or transitionTable.has_key((x, y+1))):
          frontier.push((x, y+1))
          transitionTable[(x, y+1)] = (x, y)

    ghostDistanceList = []

    for ghostPos in GhostsPos:
      count = 0
      child = ghostPos    
      while ( True ):
        parent = transitionTable.get(child)
        if parent == None:
          break
        count += 1
        child = parent

      ghostDistanceList.append(count)

    return ghostDistanceList


def GhostDistance(GameState, agentIndex):
    from util import Queue
    frontier = Queue()

    position = GameState.getPacmanPosition()
    GhostPos = GameState.getGhostPosition(agentIndex)
    frontier.push(position)
    explored = set()
    transitionTable = dict()
    count = 0
    while ( True ):
      if frontier.isEmpty():
        return float("inf")
      x, y = frontier.pop()
      if (x, y) == GhostPos:
        child = (x, y)
        while ( True ):
          parent = transitionTable.get(child)
          if parent == None:
            break
          count += 1
          child = parent
        return count
      explored.add((x, y))
      if not GameState.hasWall(x-1, y):
        if not ((x-1, y) in explored or transitionTable.has_key((x-1, y))):
          frontier.push((x-1, y))
          transitionTable[(x-1, y)] = (x, y)

      if not GameState.hasWall(x+1, y):
        if not ((x+1, y) in explored or transitionTable.has_key((x+1, y))):
          frontier.push((x+1, y))
          transitionTable[(x+1, y)] = (x, y)

      if not GameState.hasWall(x, y-1):
        if not ((x, y-1) in explored or transitionTable.has_key((x, y-1))):
          frontier.push((x, y-1))
          transitionTable[(x, y-1)] = (x, y)

      if not GameState.hasWall(x, y+1):
        if not ((x, y+1) in explored or transitionTable.has_key((x, y+1))):
          frontier.push((x, y+1))
          transitionTable[(x, y+1)] = (x, y)

def shortestCapsuleDistance(GameState):
    from util import Queue
    frontier = Queue()

    position = GameState.getPacmanPosition()
    capsulesPos = GameState.getCapsules()
    frontier.push(position)
    explored = set()
    transitionTable = dict()
    count = 0
    while ( True ):
      if frontier.isEmpty():
        return float("inf")
      x, y = frontier.pop()
      if (x, y) in capsulesPos:
        child = (x, y)
        while ( True ):
          parent = transitionTable.get(child)
          if parent == None:
            break
          count += 1
          child = parent
        return count
      explored.add((x, y))
      if not GameState.hasWall(x-1, y):
        if not ((x-1, y) in explored or transitionTable.has_key((x-1, y))):
          frontier.push((x-1, y))
          transitionTable[(x-1, y)] = (x, y)

      if not GameState.hasWall(x+1, y):
        if not ((x+1, y) in explored or transitionTable.has_key((x+1, y))):
          frontier.push((x+1, y))
          transitionTable[(x+1, y)] = (x, y)

      if not GameState.hasWall(x, y-1):
        if not ((x, y-1) in explored or transitionTable.has_key((x, y-1))):
          frontier.push((x, y-1))
          transitionTable[(x, y-1)] = (x, y)

      if not GameState.hasWall(x, y+1):
        if not ((x, y+1) in explored or transitionTable.has_key((x, y+1))):
          frontier.push((x, y+1))
          transitionTable[(x, y+1)] = (x, y)


def shortestFoodDistance(GameState):
    from util import Queue
    frontier = Queue()

    position = GameState.getPacmanPosition()
    frontier.push(position)
    explored = set()
    transitionTable = dict()
    count = 0
    while ( True ):
      if frontier.isEmpty():
        return float("inf")
      x, y = frontier.pop()
      if GameState.hasFood(x, y):
        child = (x, y)
        while ( True ):
          parent = transitionTable.get(child)
          if parent == None:
            break
          count += 1
          child = parent
        return count
      explored.add((x, y))
      if not GameState.hasWall(x-1, y):
        if not ((x-1, y) in explored or transitionTable.has_key((x-1, y))):
          frontier.push((x-1, y))
          transitionTable[(x-1, y)] = (x, y)

      if not GameState.hasWall(x+1, y):
        if not ((x+1, y) in explored or transitionTable.has_key((x+1, y))):
          frontier.push((x+1, y))
          transitionTable[(x+1, y)] = (x, y)

      if not GameState.hasWall(x, y-1):
        if not ((x, y-1) in explored or transitionTable.has_key((x, y-1))):
          frontier.push((x, y-1))
          transitionTable[(x, y-1)] = (x, y)

      if not GameState.hasWall(x, y+1):
        if not ((x, y+1) in explored or transitionTable.has_key((x, y+1))):
          frontier.push((x, y+1))
          transitionTable[(x, y+1)] = (x, y)

def shortestGhostDistance(GameState):
    from util import Queue
    frontier = Queue()

    position = GameState.getPacmanPosition()
    GhostsPos = GameState.getGhostPositions()
    frontier.push(position)
    explored = set()
    transitionTable = dict()
    count = 0
    while ( True ):
      if frontier.isEmpty():
        return float("inf")
      x, y = frontier.pop()
      if (x, y) in GhostsPos:
        child = (x, y)
        while ( True ):
          parent = transitionTable.get(child)
          if parent == None:
            break
          count += 1
          child = parent
        return count
      explored.add((x, y))
      if not GameState.hasWall(x-1, y):
        if not ((x-1, y) in explored or transitionTable.has_key((x-1, y))):
          frontier.push((x-1, y))
          transitionTable[(x-1, y)] = (x, y)

      if not GameState.hasWall(x+1, y):
        if not ((x+1, y) in explored or transitionTable.has_key((x+1, y))):
          frontier.push((x+1, y))
          transitionTable[(x+1, y)] = (x, y)

      if not GameState.hasWall(x, y-1):
        if not ((x, y-1) in explored or transitionTable.has_key((x, y-1))):
          frontier.push((x, y-1))
          transitionTable[(x, y-1)] = (x, y)

      if not GameState.hasWall(x, y+1):
        if not ((x, y+1) in explored or transitionTable.has_key((x, y+1))):
          frontier.push((x, y+1))
          transitionTable[(x, y+1)] = (x, y)

# Abbreviation
better = betterEvaluationFunction


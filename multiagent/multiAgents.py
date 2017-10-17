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

        ReturnAction = legalMoves[chosenIndex]
        "Add more of your code here if you want to"


        Nxt_GameState = gameState.generatePacmanSuccessor(ReturnAction)
        Nxt_legalMoves = [action for action in Nxt_GameState.getLegalActions() if action != 'Stop'] 
        if len(Nxt_legalMoves) == 0:
          return ReturnAction
        Nxt_scores = [self.evaluationFunction(Nxt_GameState, Nxt_action) for Nxt_action in Nxt_legalMoves]
        Nxt_bestScore = max(Nxt_scores)
        Nxt_bestIndices = [Nxt_index for Nxt_index in range(len(Nxt_scores)) if Nxt_scores[Nxt_index] == Nxt_bestScore]
        Nxt_chosenIndex = random.choice(Nxt_bestIndices) # Pick randomly among the best

        Nxt_action = Nxt_legalMoves[Nxt_chosenIndex]


        if Directions.REVERSE[ReturnAction] != Nxt_action:
          return ReturnAction
        else:
          legalMoves.remove(ReturnAction)
          scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
          bestScore = max(scores)
          bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
          chosenIndex = random.choice(bestIndices) # Pick randomly among the best
          ReturnAction = legalMoves[chosenIndex]
          return ReturnAction

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
        newPos = successorGameState.getPacmanPosition()
        currPos = currentGameState.getPacmanPosition()
        Food = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPos = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        Allcapsules = successorGameState.getCapsules()

        score = 0
        for capsule in Allcapsules:
          distance = manhattanDistance(capsule, newPos)
          if newPos == capsule:
            score += 5000
          elif distance < 1:
            score += 1000



        for index, ScaredTimer in enumerate(newScaredTimes):
          if ScaredTimer != 0:
            distance = manhattanDistance(newPos, newGhostPos[index])
            if newPos == newGhostPos[index]:
              score += 3000
            elif  distance < 1:
              score += 2500
            elif distance < 2:
              score += 1000
            elif distance < 3:
              score += 700
            elif distance*3 < ScaredTimer:
              score += 200
          else:
            if newPos == newGhostPos[index]:
              score = -999999
            elif  manhattanDistance(newPos, newGhostPos[index]) < 1:
              score -= 3000
            elif manhattanDistance(newPos, newGhostPos[index]) < 2:
              score -= 1000
            elif manhattanDistance(newPos, newGhostPos[index]) < 3:
              score -= 250

        for foodPosition in Food.asList():
          if foodPosition == newPos:
            score += 920
          else:
            score += 300/manhattanDistance(newPos, foodPosition)

        "*** YOUR CODE HERE ***"
        return score
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

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


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
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()  # action 新增了 STOP
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action) -> float:
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
        food = currentGameState.getFood().asList()  # 注意这里不要用successorGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()  # newGhostStates : list[GhostState]
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        e_val = 0
        if newPos in food:
            e_val += 100
            food.remove(newPos)
        e_val += 80 / getLeastDistance(newPos, tuple(food))
        for i in range(len(newGhostStates)):
            if newScaredTimes[i] == 0:
                if getLeastDistance(newPos, (newGhostStates[i].getPosition(),)) <= 2:
                    e_val -= 200
                else:
                    e_val -= 1 / getLeastDistance(newPos, (newGhostStates[i].getPosition(),))
        return e_val


def getLeastDistance(state: tuple[int, int], corners: tuple[tuple[int, int]]) -> int:
    """辅助判断最短到达角落的距离"""
    dist = 999999
    for corner in corners:
        if dist > abs(state[0] - corner[0]) + abs(state[1] - corner[1]):
            dist = abs(state[0] - corner[0]) + abs(state[1] - corner[1])
    return dist


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    注意这个只是pacman的代理不会代理ghost
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)  # depth 是用来控制搜索深度的,注意这里的1个depth指的是所有agent都走了一步


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState) -> str:
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

        gameState.isWin():
        Returns whether the game state is a winning state

        gameState.isLose():
        Returns whether the game state is a losing state
        """
        actions = gameState.getLegalActions(self.index)
        path = {}
        for action in actions:
            path[self.searchHelper(gameState.generateSuccessor(self.index, action), 1, 1)] = action
        return path[max(path.keys())]

    def searchHelper(self, gameState: GameState, depth: int, agentIndex: int) -> int:
        """辅助搜索函数"""
        if gameState.isWin() or gameState.isLose() or depth == self.depth * gameState.getNumAgents():
            return self.evaluationFunction(gameState)
        actions: list = gameState.getLegalActions(agentIndex)
        values = []
        for action in actions:
            values.append(self.searchHelper(gameState.generateSuccessor(agentIndex, action), depth + 1,
                                            (agentIndex + 1) % gameState.getNumAgents()))
        if agentIndex == 0:
            return max(values)
        else:
            return min(values)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        actions = gameState.getLegalActions(self.index)
        path = {}
        alpha = -9999
        beta = 9999
        for action in actions:
            value = self.searchHelper(gameState.generateSuccessor(self.index, action), 1, 1, alpha, beta)
            path[value] = action
            alpha = max(alpha, value)
        return path[alpha]

    def searchHelper(self, gameState: GameState, depth: int, agentIndex: int, alpha: int, beta: int) -> int:
        """辅助搜索函数"""
        if gameState.isWin() or gameState.isLose() or depth == self.depth * gameState.getNumAgents():
            return self.evaluationFunction(gameState)
        actions: list = gameState.getLegalActions(agentIndex)
        if agentIndex == 0:
            value = -9999
            for action in actions:
                value = max(value, self.searchHelper(gameState.generateSuccessor(agentIndex, action), depth + 1,
                                                     (agentIndex + 1) % gameState.getNumAgents(), alpha, beta))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value
        else:
            value = 9999
            for action in actions:
                value = min(value, self.searchHelper(gameState.generateSuccessor(agentIndex, action), depth + 1,
                                                     (agentIndex + 1) % gameState.getNumAgents(), alpha, beta))
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        actions = gameState.getLegalActions(self.index)
        path = {}
        for action in actions:
            path[self.searchHelper(gameState.generateSuccessor(self.index, action), 1, 1)] = action
        return path[max(path.keys())]

    def searchHelper(self, gameState: GameState, depth: int, agentIndex: int) -> float:
        """辅助搜索函数"""
        if gameState.isWin() or gameState.isLose() or depth == self.depth * gameState.getNumAgents():
            return self.evaluationFunction(gameState)
        actions: list = gameState.getLegalActions(agentIndex)
        values = []
        for action in actions:
            values.append(self.searchHelper(gameState.generateSuccessor(agentIndex, action), depth + 1,
                                            (agentIndex + 1) % gameState.getNumAgents()))
        if agentIndex == 0:
            return max(values)
        else:
            return sum(values) / len(values)


def betterEvaluationFunction(currentGameState: GameState) -> float:
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    这里要和project1的heuristic区分开来,倒是和reflexAgent的evaluationFunction很像
    DESCRIPTION: <通过综合考虑当前状态的分数，剩余食物数量，食物与pacman的距离，胶囊数量，来评估当前状态的好坏，其中当前分数是为了避免
    吃豆人徘徊或暂停不动，是一个惩罚分数，剩余食物数量是为了鼓励吃豆人吃掉食物，食物与pacman的距离是为了鼓励吃豆人吃掉距离自己最近的食物，
    胶囊数量是为了鼓励吃豆人吃掉胶囊，因为吃掉胶囊后，吃豆人可以吃掉幽灵，所以胶囊数量也是一个鼓励分数。>
    """

    if currentGameState.isWin():
        return 999999
    if currentGameState.isLose():
        return -999999
    evaluate_val = currentGameState.getScore()
    capsule_Pos: list[tuple[int, int]] = currentGameState.getCapsules()
    food_num = currentGameState.getNumFood()
    food_list = currentGameState.getFood().asList()
    pacman_Pos = currentGameState.getPacmanPosition()
    pacman_dist = getLeastDistance(pacman_Pos, tuple(food_list))
    evaluate_val = evaluate_val * 0.1 - food_num * 0.25 - pacman_dist * 0.1 - len(capsule_Pos) * 0.4
    return evaluate_val


# Abbreviation
better = betterEvaluationFunction

# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem) -> list[str]:
    """
    Search the deepest nodes in the search tree first.
=
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # 目前仅限于被positionsearchproblem调用，没有使用successor中的cost
    from util import Stack
    frontier = Stack()
    next_node = problem.getStartState()
    actions = []
    path = {next_node: None}
    node_expanded = set()
    while not problem.isGoalState(next_node):
        node_expanded.add(next_node)
        for successor in problem.getSuccessors(next_node):
            if successor[0] not in path:
                frontier.push(successor[0])
            if successor[0] not in node_expanded:
                path[successor[0]] = (next_node, successor[1])
        next_node = frontier.pop()
    while next_node != problem.getStartState():
        actions.insert(0, path[next_node][1])
        next_node = path[next_node][0]
    return actions


def breadthFirstSearch(problem: SearchProblem) -> list[str]:
    """Search the shallowest nodes in the search tree first."""
    from util import Queue
    frontier = Queue()
    next_node = (problem.getStartState(), None, 0)
    path = {next_node[0]: None}
    actions = []
    while not problem.isGoalState(next_node[0]):
        for successor in problem.getSuccessors(next_node[0]):
            if successor[0] not in path:
                frontier.push(successor)
                path[successor[0]] = next_node
        next_node = frontier.pop()
    while next_node != (problem.getStartState(), None, 0):
        actions.insert(0, next_node[1])
        next_node = path[next_node[0]]
    return actions


def uniformCostSearch(problem: SearchProblem) -> list[str]:
    """Search the node of the least total cost first."""
    from util import PriorityQueue
    frontier = PriorityQueue()
    next_node = problem.getStartState()
    path = {next_node: (None, None, 0, 1)}
    # (parent, action, cost, collected) parent:(int, int) action:str cost:int collected:int
    actions = []
    while not problem.isGoalState(next_node):
        for successor in problem.getSuccessors(next_node):
            if successor[0] not in path:  # 如果不在frontier中，就加入frontier并更新路径
                frontier.push(successor[0], path[next_node][2] + successor[2])
                path[successor[0]] = (next_node, successor[1], path[next_node][2] + successor[2], 0)
            elif not path[successor[0]][3] and path[next_node][2] + successor[2] < path[successor[0]][2]:
                # 如果在frontier中，但是没有被收集过，就视情况更新frontier和路径
                frontier.update(successor[0], path[next_node][2] + successor[2])
                path[successor[0]] = (next_node, successor[1], path[next_node][2] + successor[2], 0)
        next_node = frontier.pop()
        path[next_node] = (path[next_node][0], path[next_node][1], path[next_node][2], 1)
    while next_node != problem.getStartState():
        actions.insert(0, path[next_node][1])
        next_node = path[next_node][0]
    return actions


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

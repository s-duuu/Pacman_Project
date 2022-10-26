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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

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

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        import math

        ghost_dis_square_sum = 1
        close_ghost = 0
        food_list = newFood.asList()
        min_food_dis = float(math.inf)

        # 1st, 2nd parameter : sum of square of distance to ghosts, too close ghost
        # threshold of 2nd parameter = 5
        for ghost in newGhostStates:
            ghost_pos = ghost.getPosition()
            dis_square = util.manhattanDistance(newPos, ghost_pos) ** 2
            ghost_dis_square_sum += dis_square

            if dis_square <= 5:
                close_ghost += 1
        
        parameter_1 = float(2/ghost_dis_square_sum)
        parameter_2 = float(close_ghost)

        # 3rd parameter : minimum distance to food ()
        for food in food_list:
            distance = util.manhattanDistance(newPos, food)
            if distance < min_food_dis:
                min_food_dis = distance
        parameter_3 = float(1/min_food_dis)
        return childGameState.getScore() - parameter_1 - parameter_2 + parameter_3

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

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        import math
        pacman_index = 0
        # implementation of minimax value function
        def minimax_value(gameState, cur_agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            else:
                if cur_agentIndex == pacman_index:
                    next_agentIndex = cur_agentIndex + 1
                    return max(minimax_value(gameState.getNextState(cur_agentIndex, action), next_agentIndex, depth) for action in gameState.getLegalActions(cur_agentIndex))

                else:
                    next_agentIndex = cur_agentIndex + 1
                    if  cur_agentIndex == gameState.getNumAgents() - 1:
                        next_agentIndex = pacman_index
                        depth += 1
                    
                    return min(minimax_value(gameState.getNextState(cur_agentIndex, action), next_agentIndex, depth) for action in gameState.getLegalActions(cur_agentIndex))

        # implementation of minimax decision
        max_val = -float(math.inf)
        max_action = ""
        
        for action in gameState.getLegalActions(pacman_index):
            score = minimax_value(gameState.getNextState(pacman_index, action), 1, 0)
            if score > max_val:
                max_val = score
                max_action = action

        return max_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        import math
        pacman_index = 0
        # implementation of max-value function
        def max_value(gameState, cur_agentIndex, depth, alpha, beta):
            v = -float(math.inf)
            next_agentIndex = cur_agentIndex + 1
            for successor in gameState.getLegalActions(cur_agentIndex):
                v = max(v, decision(gameState.getNextState(cur_agentIndex, successor), next_agentIndex, depth, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        
        # implementation of min-value function
        def min_value(gameState, cur_agentIndex, depth, alpha, beta):
            next_agentIndex = cur_agentIndex + 1
            if  cur_agentIndex == gameState.getNumAgents() - 1:
                next_agentIndex = pacman_index
                depth += 1
            
            v = float(math.inf)
            for action in gameState.getLegalActions(cur_agentIndex):
                v = min(v, decision(gameState.getNextState(cur_agentIndex, action), next_agentIndex, depth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            
            return v

        # implementation of decision function for both pacman and ghosts
        def decision(gameState, agentIndex, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0:
                return max_value(gameState, agentIndex, depth, alpha, beta)
            
            else:
                return min_value(gameState, agentIndex, depth, alpha, beta)
        

        # implementation of main code
        max_result = -float(math.inf)
        max_action = "STOP"

        alpha = -float(math.inf)
        beta = float(math.inf)

        for action in gameState.getLegalActions(pacman_index):
            ghost_score = decision(gameState.getNextState(pacman_index, action), 1, 0, alpha, beta)

            if ghost_score > max_result:
                max_result = ghost_score
                max_action = action
            
            if max_result > beta:
                return max_result
            
            alpha = max(alpha, max_result)
        
        return max_action



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
        import math
        pacman_index = 0

        # implementation of expectimax function (since it is not expecti"mini"max, there's no return of minimum value)
        def expectimax_value(gameState, cur_agentIndex, depth):

            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            if cur_agentIndex == pacman_index:
                next_agentIndex = cur_agentIndex + 1
                return max(expectimax_value(gameState.getNextState(cur_agentIndex, action), next_agentIndex, depth) for action in gameState.getLegalActions(cur_agentIndex))
            
            else:
                next_agentIndex = cur_agentIndex + 1
                if cur_agentIndex == gameState.getNumAgents() - 1:
                    next_agentIndex = 0
                    depth += 1

                probability = float(1 / len(gameState.getLegalActions(cur_agentIndex)))
                return sum(expectimax_value(gameState.getNextState(cur_agentIndex, action), next_agentIndex, depth) for action in gameState.getLegalActions(cur_agentIndex)) * probability 

        # implementation of expectimax decision
        max_val = -float(math.inf)
        max_action = "STOP"
        
        for action in gameState.getLegalActions(pacman_index):
            score = expectimax_value(gameState.getNextState(pacman_index, action), 1, 0)
            if score > max_val:
                max_val = score
                max_action = action

        return max_action
        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    In pacman.py, there is a function named getCapsules() which returns list of capsules.
    Similar with getFood() function, I tried to add one more parameter into previous evaluationfunction, minimum distance to capsule.
    Also, I tried to add one more condition when dealing with distance to ghosts.
    If there is a ghost with scared time which is larger than distance to it, we can ignore the ghost.
    Following code is my implementation.
    """
    "*** YOUR CODE HERE ***"
    import math
    Pacman_pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    Capsules = currentGameState.getCapsules()
    Ghost_states = currentGameState.getGhostStates()
    Scared_times = [ghostState.scaredTimer for ghostState in Ghost_states]

    ghost_dis_square_sum = 1
    close_ghost = 0
    food_list = Food.asList()
    min_food_dis = float(math.inf)
    min_capsule_dis = float(math.inf)
    ghost_idx = 0
    ghost_num = len(Ghost_states)+1

    # 1st, 2nd parameter : sum of square of distance to ghosts, too close ghost
    # threshold of 2nd parameter = 5
    for ghost in Ghost_states:
        ghost_pos = ghost.getPosition()
        
        if Scared_times[ghost_idx] < util.manhattanDistance(Pacman_pos, ghost_pos):
            ghost_num -= 1
            dis_square = util.manhattanDistance(Pacman_pos, ghost_pos) ** 2
            ghost_dis_square_sum += dis_square

            if dis_square <= 2:
                close_ghost += 1
        ghost_idx += 1
    
    parameter_1 = float(1/ghost_dis_square_sum)
    parameter_2 = float(close_ghost)

    # 3rd parameter : minimum distance to food 
    for food in food_list:
        distance = util.manhattanDistance(Pacman_pos, food)
        if distance < min_food_dis:
            min_food_dis = distance
    parameter_3 = float(10/min_food_dis)

    # 4th parameter : minimum distance to capsule
    for capsule in Capsules:
        distance = util.manhattanDistance(Pacman_pos, capsule)
        if distance < min_capsule_dis:
            min_capsule_dis = distance
    parameter_4 = float(10/min_capsule_dis)

    parameter_5 = float(1/ghost_num)
    print("-----------------------")
    print("<Pacman> : ", Pacman_pos)
    print("1 : ", parameter_1)
    print("2 : ", parameter_2)
    print("3 : ", parameter_3)
    print("4 : ", parameter_4)
    print("5 : ", parameter_5)
    print("Total : ", currentGameState.getScore() - parameter_1 - parameter_2 + parameter_3 + parameter_4 + parameter_5)
    return currentGameState.getScore() - parameter_1 - parameter_2 + parameter_3 + parameter_4 + parameter_5

# Abbreviation
better = betterEvaluationFunction

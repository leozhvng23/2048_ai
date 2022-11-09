from BaseAI import BaseAI
from Grid import Grid
import time
from random import randint

TIME = 0.2
MONO_WEIGHT = 100000
MONO_RATIO = 0.5
CORNER_WEIGHT = 1000
WEIGHTS = [[6, 5, 4, 3], [5, 4, 3, 2], [4, 3, 2, 1], [3, 2, 1, 0]]
PATHS = [
    [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 3),
        (1, 2),
        (1, 1),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 3),
        (3, 2),
        (3, 1),
        (3, 0),
    ],
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (3, 1),
        (2, 1),
        (1, 1),
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (3, 3),
        (2, 3),
        (1, 3),
        (0, 3),
    ],
]


def getMonotonicity(grid):
    """calculates the monotonicity heuristic of the board"""
    # following 2 paths that starts at the top left corner
    # and check if the tiles are in a monotonic decreasing order
    # score is calculated as max of the sum of the linearized values on the board
    # multiplied by the values of a geometric sequence with common ratio r = 0.5.
    # credit: https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048
    weight = MONO_WEIGHT
    max_score = float("-inf")
    for path in PATHS:
        score = 0
        for pos in path:
            score += grid.getCellValue(pos) * weight
            weight *= MONO_RATIO
        max_score = max(max_score, score)
    return score


def getSmoothness(grid):
    """calculates the smoothness heuristic of the board"""
    # smoothness measures the value difference between neighboring tiles
    # credit: https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048

    score = 0
    for r in range(grid.size):
        for c in range(grid.size):
            if r < grid.size - 1:
                # smoothness for row neighbor
                score += abs(grid.getCellValue((r, c)) - grid.getCellValue((r + 1, c)))
            if c < grid.size - 1:
                # smoothness for col neighbor
                score += abs(grid.getCellValue((r, c)) - grid.getCellValue((r, c + 1)))
    return score


def getWeightedSum(grid):
    """calculates the weighted sum heuristic of the board"""
    score = 0
    for r in range(grid.size):
        for c in range(grid.size):
            value = grid.getCellValue((r, c))
            score += value * WEIGHTS[r][c]
    return score


def getMaxCorner(grid):
    """Rewards board if the max tile is in the top left corner"""
    if grid.getCellValue((0, 0)) == grid.getMaxTile():
        return CORNER_WEIGHT

    return -CORNER_WEIGHT


class IntelligentAgent(BaseAI):
    def timesup(self, cur_time):
        """Check for time limit"""
        return cur_time - self.start_time >= TIME

    def getMove(self, grid):
        """returns the best next move"""
        self.start_time = time.process_time()

        # iterative deepening search with initial depth limit of 3
        prev, self.depth = None, 2
        while True:
            self.depth += 1
            move, _ = self.maximize(grid, float("-inf"), float("inf"), 0)
            if self.timesup(time.process_time()):
                # time limit reached
                if prev is not None:
                    return prev
                else:
                    moveset = grid.getAvailableMoves()
                    return moveset[randint(0, len(moveset) - 1)][0]
            prev = move

    def chance(self, state, alpha, beta, depth):
        """chance event"""
        if self.timesup(time.process_time()):
            return 0

        # 90% chance of placing 2's
        twos = 0.9 * self.minimize(state, alpha, beta, depth, 2)
        # 10% chance of placing 4's
        fours = 0.1 * self.minimize(state, alpha, beta, depth, 4)

        return twos + fours

    def maximize(self, state, alpha, beta, depth):
        """maximizing agent: find child state with the highest utility value"""
        if self.timesup(time.process_time()):
            return None, 0

        # terminal test
        moveset = state.getAvailableMoves()
        if not moveset:
            return (None, self.evaluate(state))

        maxMove, maxUtility = None, float("-inf")
        for move, child in moveset:
            utility = self.chance(child, alpha, beta, depth + 1)
            if utility > maxUtility:
                maxMove, maxUtility = move, utility
            if maxUtility >= beta:
                break
            if maxUtility > alpha:
                alpha = maxUtility

        return maxMove, maxUtility

    def minimize(self, state, alpha, beta, depth, val):
        """minimizing agent: find child state with the lowest utility value"""
        # terminal test
        if depth >= self.depth:
            return self.evaluate(state)

        minUtility = float("inf")
        for pos in state.getAvailableCells():
            child = state.clone()
            child.setCellValue(pos, val)
            _, utility = self.maximize(child, alpha, beta, depth)
            minUtility = min(utility, minUtility)
            if minUtility <= alpha:
                break
            if minUtility < beta:
                beta = minUtility

        return minUtility

    def evaluate(self, grid):
        """evaluates the state using 4 heuristics"""

        monotonicity = getMonotonicity(grid)
        smoothness = getSmoothness(grid)
        weighted_sum = getWeightedSum(grid)
        max_corner = getMaxCorner(grid)

        return monotonicity - smoothness + weighted_sum + max_corner

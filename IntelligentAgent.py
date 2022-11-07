from BaseAI import BaseAI
from Paths import PATHS
from Grid import Grid
import math
import time
from random import randint

TIME = 0.2
CORNER_WEIGHT = 1000
# WEIGHTS = [[2048, 1024, 64, 32], [512, 128, 16, 2], [256, 8, 2, 1], [4, 2, 1, 1]]


def getMonotonicity(grid, ratio=0.9, weight=1000):
    """calculates the monotonicity heuristic of the board"""
    # following 8 paths and check if the tiles are in a monotonic decreasing order
    # score is calculated as max of the sum of the linearized values on the board
    # multiplied by the values of a geometric sequence with common ratio r = 0.25.
    # credit: https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048

    max_score = float("-inf")
    for path in PATHS:
        score = 0
        for pos in path:
            score += grid.getCellValue(pos) * weight
            weight *= ratio
        max_score = max(max_score, score)

    # print("monotonicity:", score)
    return score


def getSmoothness(grid):
    """calculates the smoothness heuristic of the board"""
    # smoothness measures the value difference between neighboring tiles
    # credit: https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048

    score = 0
    for r in range(grid.size):
        for c in range(grid.size):
            if r < grid.size - 1:
                score += abs(grid.getCellValue((r, c)) - grid.getCellValue((r + 1, c)))
            if c < grid.size - 1:
                score += abs(grid.getCellValue((r, c)) - grid.getCellValue((r, c + 1)))
    return score


# def getWeightedSum(grid):
#     score = 0
#     for r in range(grid.size):
#         for c in range(grid.size):
#             value = grid.getCellValue((r, c))
#             score += value * WEIGHTS[r][c]
#     return score


def getMaxCorner(grid):
    max_val = grid.getMaxTile()
    for r, c in [(0, 0), (0, 3), (3, 0), (3, 3)]:
        if grid.getCellValue((r, c)) == max_val:
            return math.log2(CORNER_WEIGHT * max_val)

    return -math.log2(CORNER_WEIGHT * max_val)


class IntelligentAgent(BaseAI):
    def timesup(self, cur_time):
        return cur_time - self.prevtime >= TIME

    def getMove(self, grid):
        # Selects a random move and returns it
        # moveset = grid.getAvailableMoves()
        # return random.choice(moveset)[0] if moveset else None
        self.prevtime = time.process_time()

        prev, self.depth = None, 2

        while True:
            self.depth += 1
            move, _ = self.maximize(grid, float("-inf"), float("inf"), 0)
            if self.timesup(time.process_time()):
                if prev is not None:
                    return prev
                else:
                    moveset = grid.getAvailableMoves()
                    return moveset[randint(0, len(moveset) - 1)][0]
            prev = move

    def chance(self, state, alpha, beta, depth):
        if self.timesup(time.process_time()):
            return 0

        # 90% chance of placing 2's
        twos = 0.95 * self.minimize(state, alpha, beta, depth, 2)

        # 10% chance of placing 4's
        fours = 0.05 * self.minimize(state, alpha, beta, depth, 4)

        return twos + fours

    def maximize(self, state, alpha, beta, depth):
        if self.timesup(time.process_time()):
            return None, 0

        moveset = state.getAvailableMoves()
        if not moveset:
            return (None, float("-inf"))

        maxMove, maxUtility = None, float("-inf")

        for pair in moveset:
            move, child = pair
            utility = self.chance(child, alpha, beta, depth + 1)
            if utility > maxUtility:
                maxMove, maxUtility = move, utility
            if maxUtility >= beta:
                break
            if maxUtility > alpha:
                alpha = maxUtility

        return maxMove, maxUtility

    def minimize(self, state, alpha, beta, depth, val):
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
        """evaluates the grid using heuristics"""
        # w1, w2, w3, w4, w5 = 3, 3, 2, 4, 1
        w1, w2, w3, w4, w5 = 10, 5, 15, 1, 10
        monotonicity = getMonotonicity(grid)
        smoothness = getSmoothness(grid)
        # weighted_sum = getWeightedSum(grid)
        max_tile = grid.getMaxTile()
        max_corner = getMaxCorner(grid)
        # weighted_sum = getWeightedSum(grid)

        # score = (
        #     (w1 * monotonicity)
        #     - (w2 * math.log2(smoothness))
        #     + (w3 * math.log2(max_tile))
        #     + (w4 * empty_cells)  w
        #     + (w5 * math.log2(max_corner))
        # )
        # score = (
        #     1000
        #     + (w1 * monotonicity)
        #     - (w2 * math.log2(smoothness))
        #     + (w3 * math.log2(max_tile))
        #     + (w4 * empty_cells)
        #     + (w5 * math.log2(max_corner))
        # )
        score = (
            (w1 * monotonicity)
            - (w2 * math.log2(smoothness))
            + (w3 * math.log2(max_tile))
            + (w5 * max_corner)
        )
        # print(monotonicity, -smoothness, max_corner)
        # print(
        #     w1 * monotonicity,
        #     (w2 * math.log2(smoothness)),
        #     (w3 * math.log2(max_tile)),
        #     w5 * max_corner,
        # )
        # print(score)

        return score


# if __name__ == "__main__":
#     g1 = Grid()
#     g1.map = [[0, 4, 2, 4], [0, 8, 4, 2], [4, 32, 64, 8], [16, 4, 8, 4]]
#     ai = IntelligentAgent()
#     ai.evaluate(g1)

#     g2 = Grid()
#     g2.map = [[8, 32, 64, 512], [4, 8, 16, 256], [2, 4, 8, 32], [0, 0, 4, 8]]
#     ai = IntelligentAgent()
#     moves = len(g2.getAvailableMoves())
#     ai.evaluate(g2)

#     g2 = Grid()
#     g2.map = [[1024, 4, 64, 0], [4, 8, 32, 4], [4, 256, 512, 8], [64, 4, 8, 2]]
#     ai = IntelligentAgent()
#     ai.evaluate(g2)

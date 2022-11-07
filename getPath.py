from enum import Enum
import collections

SZ = 4  # 4x4 2048 board size


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


def getNextRight(y, x, majorDir, minorDir):
    """
    Get the coordinate of the next tile
    if the main direction is right
    """
    x += 1
    if x == SZ:
        y = (y + 1) if minorDir == Direction.DOWN else (y - 1)
        x = SZ - 1
        majorDir = Direction.LEFT
    return y, x, majorDir


def getNextLeft(y, x, majorDir, minorDir):
    """
    Get the coordinate of the next tile
    if the main direction is left
    """
    x -= 1
    if x == -1:
        y = (y + 1) if minorDir == Direction.DOWN else (y - 1)
        x = 0
        majorDir = Direction.RIGHT
    return y, x, majorDir


def getNextUp(y, x, majorDir, minorDir):
    """
    Get the coordinate of the next tile
    if the main direction is up
    """
    y -= 1
    if y == -1:
        x = (x + 1) if minorDir == Direction.RIGHT else (x - 1)
        y = 0
        majorDir = Direction.DOWN
    return y, x, majorDir


def getNextDown(y, x, majorDir, minorDir):
    """
    Get the coordinate of the next tile
    if the main direction is down
    """
    y += 1
    if y == SZ:
        x = (x + 1) if minorDir == Direction.RIGHT else (x - 1)
        y = SZ - 1
        majorDir = Direction.UP
    return y, x, majorDir


def getNextTile(y, x, majorDir, minorDir):
    """
    Get the coordinate of the next tile depending of the directions.
    """
    if majorDir == Direction.RIGHT:
        y, x, majorDir = getNextRight(y, x, majorDir, minorDir)
    elif majorDir == Direction.LEFT:
        y, x, majorDir = getNextLeft(y, x, majorDir, minorDir)
    elif majorDir == Direction.UP:
        y, x, majorDir = getNextUp(y, x, majorDir, minorDir)
    else:
        y, x, majorDir = getNextDown(y, x, majorDir, minorDir)
    return (y, x), majorDir, minorDir


def getPath(tile, majorDir, minorDir):
    """
    Get path to browse all the tiles depending of the directions.
    """
    path = []
    for _ in range(SZ * SZ):
        path.append(tile)
        tile, majorDir, minorDir = getNextTile(tile[0], tile[1], majorDir, minorDir)
    return path


data = [
    [(0, 0), Direction.RIGHT, Direction.DOWN],
    [(0, 0), Direction.DOWN, Direction.RIGHT],
    [(0, 3), Direction.LEFT, Direction.DOWN],
    [(0, 3), Direction.DOWN, Direction.LEFT],
    [(3, 0), Direction.RIGHT, Direction.UP],
    [(3, 0), Direction.UP, Direction.RIGHT],
    [(3, 3), Direction.LEFT, Direction.UP],
    [(3, 3), Direction.UP, Direction.LEFT],
]

# Generate the file path.go
print("package bot\n")
print("// auto-generated file by `path.py`.")
print("// Do not update it !\n")

print("var paths = [][]Tile{")
for d in data:
    print("{", end="", sep="")
    path = getPath(d[0], d[1], d[2])
    for p in path:
        print("{", p[0], ",", p[1], "},", end="", sep="")
    print("},")
print("}")

res = []
for d in data:
    res.append(getPath(d[0], d[1], d[2]))
for v in res:
    print(v)

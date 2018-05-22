
#author1:
#author2:

from grid import *
from visualizer import *
import numpy as np
import threading
from queue import PriorityQueue
import math
import cozmo
import copy

def reconstructPath(cameFromMap, final, start):
    path = [final]
    current = final
    while not (current[0] == start [0] and current[1] == start[1]):
        current = cameFromMap[current[0]][current[1]]
        path.insert(0, current)
    return path

def astar(grid, heuristic):
    """Perform the A* search algorithm on a defined grid

        Arguments:
        grid -- CozGrid instance to perform search on
        heuristic -- supplied heuristic function
    """
    goal = grid.getGoals()[0]
    startCell = grid.getStart()
    grid.clearVisited()

    residualEstimate = heuristic(startCell, goal)
    currentWeight = 0

    cameFrom = np.zeros((grid.width, grid.height)).tolist()

    goToScore = (np.ones((grid.width, grid.height)) * float("inf")).tolist()
    goToScore[startCell[0]][startCell[1]] = currentWeight

    frontier = {}
    frontier[tuple(startCell)] = residualEstimate

    while len(frontier.keys()) > 0:
        currentCell = sorted(frontier.items(), key =lambda x:x[1])[0][0]
        frontier.pop(currentCell)
        grid.addVisited(currentCell)
        
        if(currentCell[0] == goal[0] and currentCell[1] == goal[1]):
            grid.setPath(reconstructPath(cameFrom, currentCell, startCell))
            return

        for neighborData in grid.getNeighbors(currentCell):
            neighbor = neighborData[0]
            weight = neighborData[1]

            if neighborData[0] in grid.getVisited():
                continue
            

            tentative_goToScore = goToScore[currentCell[0]][currentCell[1]] + weight
            if tentative_goToScore > goToScore[neighbor[0]][neighbor[1]]:
                continue
            
            # if we got here, we are on the best path to the neighbor node so far
            cameFrom[neighbor[0]][neighbor[1]] =  currentCell
            goToScore[neighbor[0]][neighbor[1]] = tentative_goToScore
            frontierScore = goToScore[neighbor[0]][neighbor[1]] + heuristic(neighbor, goal)
            frontier[tuple(neighbor)] = frontierScore
pass 


def heuristic(current, goal):
    """Heuristic function for A* algorithm

        Arguments:
        current -- current cell
        goal -- desired goal cell
    """
    curr_x = current[0]
    curr_y = current[1]
    goal_x = goal[0]
    goal_y = goal[1]
    # simple manhattan distance. an admissible heuristic for instances where we cannot move diagonally
    return math.sqrt((curr_y - goal_y)**2 + (curr_x - goal_x)**2)


def cozmoBehavior(robot: cozmo.robot.Robot):
    """Cozmo search behavior. See assignment description for details

        Has global access to grid, a CozGrid instance created by the main thread, and
        stopevent, a threading.Event instance used to signal when the main thread has stopped.
        You can use stopevent.is_set() to check its status or stopevent.wait() to wait for the
        main thread to finish.

        Arguments:
        robot -- cozmo.robot.Robot instance, supplied by cozmo.run_program
    """
        
    global grid, stopevent
    
    while not stopevent.is_set():
        pass # Your code here


######################## DO NOT MODIFY CODE BELOW THIS LINE ####################################


class RobotThread(threading.Thread):
    """Thread to run cozmo code separate from main thread
    """
        
    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        cozmo.run_program(cozmoBehavior)


# If run as executable, start RobotThread and launch visualizer with empty grid file
if __name__ == "__main__":
    global grid, stopevent
    stopevent = threading.Event()
    grid = CozGrid("emptygrid.json")
    visualizer = Visualizer(grid)
    updater = UpdateThread(visualizer)
    updater.start()
    robot = RobotThread()
    robot.start()
    visualizer.start()
    stopevent.set()


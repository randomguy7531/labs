
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


def robot_pose_to_coords(pose, grid):
    scale = grid.scale
    offset = scale / 2
    return (int((pose.position.x + offset) / scale), int((pose.position.y + offset) / scale))

def is_cube_found(cube):
    return not (cube.pose.position.x == 0 and cube.pose.position.y == 0 and cube.pose.position.z == 0)

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
    print("starting cozmo behavior")
    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    while not stopevent.is_set():
        
        print("entering loop")
        # initially, clear the grid and initialize the robot's position
        grid.clearObstacles()
        grid.setStart(robot_pose_to_coords(robot.pose, grid))    
        # get the cube locations
        print("getting cubes")
        cubes = [robot.world.get_light_cube(cozmo.objects.LightCube1Id), robot.world.get_light_cube(cozmo.objects.LightCube2Id), robot.world.get_light_cube(cozmo.objects.LightCube3Id)]
        desired_angle = cubes[0].pose.rotation.angle_z.degrees

        # Fill in obstacles based on the non-goal cubes
        if is_cube_found(cubes[1]):
            print("found cube 1")
            grid.addObstacles([robot_pose_to_coords(cubes[1].pose, grid)])
        if is_cube_found(cubes[2]):
            print("found cube 2")
            grid.addObstacles([robot_pose_to_coords(cubes[2].pose, grid)])
            

        # If we found the goal cube, mark it as the goal
        if is_cube_found(cubes[0]):
            print("found cube  (goal)")
            grid.clearGoals()
            target_coords = robot_pose_to_coords(cubes[0].pose, grid)
            print("goal cube at {0}".format(target_coords))
            approach_x = approach_y = 0
            grid.addObstacles([target_coords])
            approach_angle = cubes[0].pose.rotation.angle_z
            approach_x = round(math.cos(approach_angle.radians), 0)
            approach_y = round(math.sin(approach_angle.radians), 0)
            
            print("adding goal at {0}".format((target_coords[0] - approach_x, target_coords[1] - approach_y)))
            grid.addGoal((target_coords[0] - approach_x, target_coords[1] - approach_y))
        # If we are at the center of the map, turn in place to look for the cube in question
        elif grid.getStart() == (int(grid.width/2), int(grid.height/2)):
            print("turning to look for goal cube")
            robot.turn_in_place(cozmo.util.radians(10))
            continue
        # If we dont see the goal cube, and we are somewhere besides the center of the map,
        # set the grid center as the desired goal
        else:
            print("moving to middle of grid")
            grid.addGoal((int(grid.width/2), int(grid.height/2)))

        # Run A* with the collected goal and obstacle info
        print("running astar")
        astar(grid, heuristic)

        # Move along the path
        print("moving along path {0}".format(grid.getPath()))
        for i in range(len(grid.getPath()) - 1):
            dx = (grid.getPath()[i + 1][0] - grid.getPath()[i][0]) * grid.scale
            dy = (grid.getPath()[i + 1][1] - grid.getPath()[i][1]) * grid.scale
            dh = cozmo.util.radians(math.atan2(dy, dx)) - robot.pose.rotation.angle_z
            robot.turn_in_place(dh).wait_for_completed()
            robot.drive_straight(cozmo.util.distance_mm(math.sqrt(dx*dx + dy*dy)), cozmo.util.speed_mmps(25)).wait_for_completed()
        #turn the correct direction
        robot.turn_in_place(cozmo.util.degrees(desired_angle - robot.pose.rotation.angle_z.degrees)).wait_for_completed()

        #if we are at the cube (with some allowed distance), turn the correct direction and stop
        if math.sqrt((robot.pose.position.x - cubes[0].pose.position.x)**2 + (robot.pose.position.y - cubes[0].pose.position.y)**2) < 300:
            print("facing goal then stopping")
            robot.turn_in_place(cozmo.util.degrees(desired_angle - robot.pose.rotation.angle_z.degrees)).wait_for_completed()
            stopevent.set()


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


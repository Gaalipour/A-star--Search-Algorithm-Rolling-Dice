"""
File name: rdmaze.py
Language: Python3
Authors:
    Ghodratollah Aalipour and Akash Venkatachalam
Description:
The objective is to implement  A* search using different heuristics and solve the rolling-die mazes. We consider each
position on maze as a (x,y)  point on the rectangular coordinate system. The dice is six-sided with all opposite die faces
summing up to 7. The initial state of our dice will have the orientation of ‘1’ on top, ‘2’ facing north and ‘3’ facing east.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import sys

def maze(file_name):
    """
    Function to open and read in the maze file and store them as numpy 2D array.
    :param file_name: maze file
    :return: the 2D numpy array obtained from input maze file.
    """
    fh = open(file_name)
    mazefile = fh.read()                                                    # Reading in the file
    lines = [[char for char in line] for line in mazefile.split("\n")]      # Spitting it and storing as a 2D array
    return (np.array(lines))

def print_dice(dice):
    """
    Function to print the orientation of the dice and its (x,y) position on the maze
    This function will also print the maze with dice on it.
    :param dice: The dictionary containing top, north, east, row and column as keys
    """
    updated_maze = np.array(maze)
    r = dice["row"]
    c = dice["column"]
    updated_maze[r,c] = dice["top"]     # Updated maze after every move made by the dice
    for line in updated_maze:
        print(" ".join(line))           # To print the maze with dice from the 2D numpy array
    print("The dice faces: ",           # Printing dice orientation and position
          "Row:", dice["row"], " Column:", dice["column"], " Top:", dice["top"], " North:", dice["north"], " East:", dice["east"])
    print()

def start_goal_entries(maze):
    """
    Function to record the start and goal location for the given maze
    :param maze: The 2D numpy array containing the maze
    :return: Start and the goal coordinates
    """
    m = len(maze)
    n = len(maze[0])
    start_loc = None
    goal_loc = None
    for i in range(0,m):
        for j in range(0,n):
            if maze[i,j] == "S":       # Searches for the string 'S'
                start_loc = (i,j)
            if maze[i,j] == "G":       # Searches for the string 'G'
                goal_loc = (i,j)
    return start_loc, goal_loc


def neighbors(dice):
    """
    Function to get all the neighbours of the given orientation of a dice
    :param dice: The dictionary containing top, north, east, row and column as keys
    :return: List of all the neighbours of the given orientation of a dice
    """
    neighbors_list = []
    r = dice["row"]
    c = dice["column"]
    m,n = maze.shape    # To get the dimension of the maze

    for pair in [(r,c+1),(r,c-1),(r-1,c),(r+1,c)]:
        if math.floor(pair[0]/m) != 0 or math.floor(pair[1]/n) != 0:   # If the neighbour is out of the maze
            continue
        elif maze[pair[0],pair[1]] == "*":      # If the neighbour is an obstacle (*)
            continue
        else:
            neighbors_list.append(pair)         # If it is a potential neighbour, then adding it to the list
    return neighbors_list

def roller(dice, i, j):
    """
    Function to roll on the dice to its neighbours and create new dict/dice/state with every possible roll.
    :param dice: The dictionary containing top, north, east, row and column as keys
    :param i: The next row value for the current dice
    :param j: The next column value for the current dice
    :return: New dictionary or new dice state
    """
    r = dice["row"]
    c = dice["column"]
    new_dice = {}

    if i == r and j == c + 1:   # roll to right
        new_dice["top"] = 7 - dice["east"]
        new_dice["north"] = dice["north"]
        new_dice["east"] = dice["top"]
        new_dice["row"] = i
        new_dice["column"] = j
        new_dice["cost"] = dice["cost"] + 1
        new_dice["parent"] = dice

    if i == r and j == c-1:      # roll to left
        new_dice["top"] = dice["east"]
        new_dice["north"] = dice["north"]
        new_dice["east"] = 7 - dice["top"]
        new_dice["row"] = i
        new_dice["column"] = j
        new_dice["cost"] = dice["cost"] + 1
        new_dice["parent"] = dice

    if i == r+1 and j == c:      # rolls to south
        new_dice["top"] = dice["north"]
        new_dice["east"] = dice["east"]
        new_dice["north"] = 7 - dice["top"]
        new_dice["row"] = i
        new_dice["column"] = j
        new_dice["cost"] = dice["cost"] + 1
        new_dice["parent"] = dice

    if i == r-1 and j == c :     # rolls to north
        new_dice["top"] = 7 - dice["north"]
        new_dice["east"] = dice["east"]
        new_dice["north"] = dice["top"]
        new_dice["row"] = i
        new_dice["column"] = j
        new_dice["cost"] = dice["cost"] + 1
        new_dice["parent"] = dice

    if new_dice["top"] == 6:      # Invalid move
        return "This is not a valid move: SIX on TOP"
    else:
        return new_dice

def evaluation(dice, heuristic):
    """
    The evaluation function, f(x) = g(x) + h(x) where g(x) is the running cost and h(x) is heuristic value
    :param dice: The dictionary containing top, north, east, row and column as keys
    :param heuristic: The name of the heuristic to call
    :return:
    """
    return dice["cost"] + heuristic(dice)

def constant_one(dice):
    """
    One type of heuristic function which always returns a constant. We can allow the constant to be zero (Dijkstra’s Algorithm) or to be one.
    For the sake of admissibility, we only allow these two values for this constant heuristic.
    :param dice: The dictionary containing top, north, east, row and column as keys
    :return: constant 0 or 1
    """
    i = dice["row"]
    j = dice["column"]
    if goal[0] == i and goal[1] == j:  # When at the goal state
        return 0
    else:
        return 1

def manhattan(dice):
    """
    One type of heuristic function which is the sum of the absolute value of differences of x’s values
    plus the sum of the absolute value of the y’s value
    :param dice: The dictionary containing top, north, east, row and column as keys
    :return: the distance vaule
    """
    i = dice["row"]
    j = dice["column"]
    distance = abs(goal[0] - i) + abs(goal[1] - j)
    return distance

def euclidean(dice):
    """
    Another type of heuristic function which is the diagonal distance; so always less than or equal to Manhattan
    :param dice: The dictionary containing top, north, east, row and column as keys
    :return: The calculated distance value
    """
    i = dice["row"]
    j = dice["column"]
    distance = np.sqrt((goal[0] - i)**2 + (goal[1] - j)**2 )
    return distance

def advaned_manhattan(dice):
    """
    Another type of heuristic function which is a modification of the Manhattan distance
    :param dice: The dictionary containing top, north, east, row and column as keys
    :return: The calculated distance value.
    """
    i = dice["row"]
    j = dice["column"]
    if i == goal[0] and j == goal[1]:
        return 0
    elif (goal[0] == i and goal[1]!= j) or (goal[0] != 0 and goal[1] == j):
        return manhattan(dice) + 2
    else:
        return manhattan(dice) + 2

def goal_test(dice):
    """
    Function to check if goal state is reached with '1' on the top of dice or not
    :param dice: The dictionary containing top, north, east, row and column as keys
    :return: True if goal state is reached with '1' on the top else False
    """
    r = dice["row"]
    c = dice["column"]
    if r == goal[0] and c == goal[1] and dice["top"] == 1:
        return True
    else:
        return False


def prioritizer(list_of_dices, new_dice, heuristic):
    """
    Function to implement the priority queue and use it in this program
    :param list_of_dices: List of dices present in the queue already
    :param new_dice: The dictionary containing top, north, east, row and column as keys
    :param heuristic: The name of the heuristic
    :return: The updated priority queue
    """
    copy = list(list_of_dices)
    copy.append(new_dice)
    new_list = sorted(copy, key = lambda node: evaluation(node, heuristic))
    return new_list

def dice_translator(dice):
    """
    Function to translate the dice dictionary into string values without the use of cost
    :param dice: The dictionary containing top, north, east, row and column as keys
    :return: A string type of dice's top, north, east, row, column
    """
    return str(dice["top"]) + str(dice["north"]) + str(dice["east"]) + str(dice["row"]) + str(dice["column"] )

def a_star_search(start, heuristic):
    """
    Function to implement the A* search
    :param start: A dice dictionary with initialized start orientation
    :param heuristic: The name of the heuristic
    """
    expanded = set()            # Set to keep track of the expanded nodes
    priority = [start]          # Initializing the priority queue
    no_nodes_priority = 1       # Keeping track of the number of nodes in priority queue
    no_nodes_taken_off = 0      # Keeping track of the number of nodes takes off the queue
    count = 0
    while len(priority) > 0 :
        dice = priority.pop(0)
        no_nodes_taken_off += 1
        expanded.add(dice_translator(dice))
        parent_dict = dice                  # Keeping track of the parent node
        count += 1
        if goal_test(dice):
            return dice, parent_dict, no_nodes_priority, no_nodes_taken_off     # Return if the goal test is True
        else:
            children = [roller(dice, pair[0], pair[1]) for pair in neighbors(dice) if not isinstance(roller(dice, pair[0], pair[1]), str)]
            for child in children:
                if dice_translator(child) not in expanded:
                    priority = list(prioritizer(priority, child, heuristic))
                    no_nodes_priority += 1
    dice = "Failed"
    parent_dict = "Failed"
    return dice, parent_dict, no_nodes_priority, no_nodes_taken_off   # If the priority queue runs out of items, return "Failed"

class parent_path_recursion():
    def __init__(self):
        """
        Function to initialize the lists to store parent connections
        """
        self.list_of_rowncol = []
        self.ls_of_dicts = []

    def parent_path(self,parent_dict):
        """
        Function to recursively backtrack the parent from the obtained goal state
        :param parent_dict: The dictionary containing top, north, east, row, column and parent as keys
        :return: The list of dictionaries and list of solution path
        """
        if parent_dict["cost"] == 0:                        # Base case for the inner most dictionary
            self.ls_of_dicts.insert(0, parent_dict)
            self.list_of_rowncol.append(str(parent_dict["row"])+str(parent_dict["column"]))
            return self.ls_of_dicts, self.list_of_rowncol
        else:
            self.ls_of_dicts.insert(0, parent_dict)
            return self.parent_path(parent_dict["parent"])  # Recursively calling the dictionary of dictionary to backtrack


########################################################################################################

file_name = sys.argv[1] # Name of the input file
maze = maze(file_name)  # Function call to maze function
start, goal = start_goal_entries(maze)
dice = {"top": 1, "north": 2, "east": 3, "row": start[0], "column": start[1], "cost": 0}
print("Initial Maze and Dice Orientation")
print_dice(dice)  # Printing the initial orientation of dice and maze


def main(heuristic):
    """
    Main Function to call the A* search
    :param heuristic: The name of heuristic function
    :return: Number of nodes in priority queue and nodes taken off the queue
    """
    my_algo = a_star_search(dice, heuristic)
    result, parent_dict, no_nodes_priority, no_nodes_taken_off = my_algo

    if result == "Failed":
        print("No Solution")
        print()
        print("#########################################")
        print("\nSearch Failed\n")
        print("\nNumber of moves in the solution: -1")
        print("Number of Nodes Put on Frontier Queue: ", no_nodes_priority)
        print("Number of Visited (Expanded): ", no_nodes_taken_off)
        return no_nodes_priority, no_nodes_taken_off

    else:
        print()
        print("#########################################")
        print("\nGoal reached with this process ....\n")
        r = parent_path_recursion()
        ls_of_dicts, list_of_rowncol = r.parent_path(parent_dict)
        for item in ls_of_dicts:
            print_dice(item)
        print("\nNumber of moves in the solution: ", len(ls_of_dicts))
        print("Number of Nodes Put on Frontier Queue: ", no_nodes_priority)
        print("Number of Visited (Expanded): ", no_nodes_taken_off)
        return no_nodes_priority, no_nodes_taken_off


heuristic_list = {constant_one:"Constant Function 1", euclidean:"Euclidean Distance",
                  manhattan:"Manhattan Distance", advaned_manhattan:"Advanced Manhattan Distance"}
performance = {}
for key, val in heuristic_list.items():
    heuristic = key
    print("--------------------------------------------------------------------")
    print("--------------------------------------------------------------------")

    print("Our heuristic: ",val)
    metrics = main(heuristic)
    performance[heuristic] = (heuristic_list[key], metrics)

data = performance.values()     # Keeping track of performance to plot the graph

print(data)

# data to plot
n_groups = 4
in_priority = [item[1][0] for item in data]
taken_off = [item[1][1] for item in data]

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, in_priority, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Total Number of Nodes in Priority')

rects2 = plt.bar(index + bar_width, taken_off, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Number of Nodes Visited')

plt.xlabel('Heuristic', fontsize = 16)
plt.ylabel('Number of Nodes', fontsize = 16)
plt.title('Performance vs Heuristic', fontsize = 18)
plt.xticks(index + bar_width, heuristic_list.values())
plt.legend()

plt.tight_layout()
plt.show()

########################################################################################################

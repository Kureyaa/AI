import sys
from collections import deque
import heapq
import math

# Read map.txt file
def read_map_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != ""]
        
    #Make lines into integers
    rows, cols = map(int, lines[0].split())
    start_row, start_col = map(int, lines[1].split())
    goal_row, goal_col = map(int, lines[2].split())
    
    # convert from 1-based to 0-based indexing
    start = (start_row - 1, start_col - 1)
    goal = (goal_row - 1, goal_col - 1)
    
    # Append rows into grid
    grid = []
    for i in range(3, 3 + rows):
        tokens = lines[i].split()
        row = []
        for token in tokens:
            if token == "X":
                row.append("X")
            else:
                row.append(int(token))
        grid.append(row)

    return rows, cols, start, goal, grid

def in_bounds(pos, rows, cols):
    r, c = pos
    return 0 <= r < rows and 0 <= c < cols

def is_obstacle(pos, grid):
    r, c = pos
    return grid[r][c] == "X"

def manhattan(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def euclidean(pos, goal):
    return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

def get_neighbors(pos, rows, cols, grid):
    r, c = pos
    # up, down, left, right
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    
    neighbors = []
    # if current position is not an obstacle and in bounds, add to neighbors
    for nr, nc in candidates:
        position = (nr, nc)
        if in_bounds(position, rows, cols) and not is_obstacle(position, grid):
            neighbors.append(position)
    return neighbors

def move_cost(current, next, grid):
    curr_height = grid[current[0]][current[1]]
    next_height = grid[next[0]][next[1]]
    
    if next_height > curr_height:
        return 1 + (next_height - curr_height)
    else:
        return 1

def reconstruct_path(parent, goal):
    if goal not in parent:
        return None  
    
    path = []
    current = goal
    
    while current is not None:
        path.append(current)
        current = parent[current]
        
    path.reverse()
    return path    

def bfs(rows, cols, start, goal, grid):
    queue = deque()
    queue.append(start)

    visited = set()
    visited.add(start)

    parent = {start: None}

    visits = create_matrix(rows, cols, 0)
    first_visit = create_matrix(rows, cols, None)
    last_visit = create_matrix(rows, cols, None)
    time = 0

    while queue:
        current = queue.popleft()

        time += 1
        r, c = current
        visits[r][c] += 1
        if first_visit[r][c] is None:
            first_visit[r][c] = time
        last_visit[r][c] = time

        if current == goal:
            return reconstruct_path(parent, goal), visits, first_visit, last_visit

        for nbr in get_neighbors(current, rows, cols, grid):
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = current
                queue.append(nbr)

    return None, visits, first_visit, last_visit

def ucs(rows, cols, start, goal, grid):
    queue = []
    heapq.heappush(queue, (0, 0, start))

    parent = {start: None}
    best_cost = {start: 0}
    counter = 0

    visits = create_matrix(rows, cols, 0)
    first_visit = create_matrix(rows, cols, None)
    last_visit = create_matrix(rows, cols, None)
    time = 0

    while queue:
        current_cost, _, current = heapq.heappop(queue)

        time += 1
        r, c = current
        visits[r][c] += 1
        if first_visit[r][c] is None:
            first_visit[r][c] = time
        last_visit[r][c] = time

        if current == goal:
            return reconstruct_path(parent, goal), visits, first_visit, last_visit

        if current_cost > best_cost[current]:
            continue

        for nbr in get_neighbors(current, rows, cols, grid):
            new_cost = current_cost + move_cost(current, nbr, grid)

            if nbr not in best_cost or new_cost < best_cost[nbr]:
                best_cost[nbr] = new_cost
                parent[nbr] = current
                counter += 1
                heapq.heappush(queue, (new_cost, counter, nbr))

    return None, visits, first_visit, last_visit

def astar(rows, cols, start, goal, grid, heuristic_name):
    queue = []

    if heuristic_name == "manhattan":
        h_start = manhattan(start, goal)
    else:
        h_start = euclidean(start, goal)

    heapq.heappush(queue, (h_start, 0, 0, start))

    parent = {start: None}
    best_cost = {start: 0}
    counter = 0

    visits = create_matrix(rows, cols, 0)
    first_visit = create_matrix(rows, cols, None)
    last_visit = create_matrix(rows, cols, None)
    time = 0

    while queue:
        _, _, current_g, current = heapq.heappop(queue)

        time += 1
        r, c = current
        visits[r][c] += 1
        if first_visit[r][c] is None:
            first_visit[r][c] = time
        last_visit[r][c] = time

        if current == goal:
            return reconstruct_path(parent, goal), visits, first_visit, last_visit

        if current_g > best_cost[current]:
            continue

        for nbr in get_neighbors(current, rows, cols, grid):
            new_g = current_g + move_cost(current, nbr, grid)

            if heuristic_name == "manhattan":
                h = manhattan(nbr, goal)
            else:
                h = euclidean(nbr, goal)

            f = new_g + h

            if nbr not in best_cost or new_g < best_cost[nbr]:
                best_cost[nbr] = new_g
                parent[nbr] = current
                counter += 1
                heapq.heappush(queue, (f, counter, new_g, nbr))

    return None, visits, first_visit, last_visit

def make_path(grid, path):
    if path is None:
        return None

    output = []
    path_set = set(path)

    for r in range(len(grid)):
        row = []
        for c in range(len(grid[0])):
            if (r, c) in path_set:
                row.append("*")
            else:
                row.append(str(grid[r][c]))
        output.append(row)

    return output

def print_grid(grid):
    for row in grid:
        print(" ".join(row))

def create_matrix(rows, cols, value):
    return [[value for _ in range(cols)] for _ in range(rows)]

def print_debug_matrix(matrix, grid):
    for r in range(len(matrix)):
        row = []
        for c in range(len(matrix[0])):
            if grid[r][c] == "X":
                row.append("X")
            else:
                val = matrix[r][c]
                if val is None or val == 0:
                    row.append(".")
                else:
                    row.append(str(val))
        print(" ".join(row))
        
        
def main():
    mode = sys.argv[1].strip().lower()
    map_file = sys.argv[2].strip()
    algorithm = sys.argv[3].strip().lower()
    heuristic = sys.argv[4].strip().lower()

    rows, cols, start, goal, grid = read_map_file(map_file)

    if algorithm == "bfs":
        path, visits, first_visit, last_visit = bfs(rows, cols, start, goal, grid)
    elif algorithm == "ucs":
        path, visits, first_visit, last_visit = ucs(rows, cols, start, goal, grid)
    elif algorithm == "astar":
        path, visits, first_visit, last_visit = astar(rows, cols, start, goal, grid, heuristic)

    if mode == "release":
        if path is None:
            print("null")
        else:
            output_grid = make_path(grid, path)
            print_grid(output_grid)

    elif mode == "debug":
        print("path:")
        if path is None:
            print("null")
        else:
            output_grid = make_path(grid, path)
            print_grid(output_grid)

        print("#visits:")
        print_debug_matrix(visits, grid)

        print("first visit:")
        print_debug_matrix(first_visit, grid)

        print("last visit:")
        print_debug_matrix(last_visit, grid)
        
main()
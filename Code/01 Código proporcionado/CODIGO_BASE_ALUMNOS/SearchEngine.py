# Required imports
import numpy as np
import networkx as nx
from Boundaries import Boundaries
from Map import EPSILON
from Formulas import euclidean_dist, manhattan_dist

# Number of nodes expanded in the heuristic search (stored in a global variable to be updated from the heuristic functions)
NODES_EXPANDED = 0


def h1(current_node, objective_node) -> np.float32:
    """ First heuristic to implement """
    global NODES_EXPANDED
    h = 0
    current_lat = current_node[1]
    current_lon = current_node[0]
    objective_lat = objective_node[1]
    objective_lon = objective_node[0]

    h = EPSILON * euclidean_dist(current_lon, current_lat, objective_lon, objective_lat)
    NODES_EXPANDED += 1
    return h

def h2(current_node, objective_node) -> np.float32:
    """ Second heuristic to implement """
    global NODES_EXPANDED
    h = 0
    current_lat = current_node[1]
    current_lon = current_node[0]
    objective_lat = objective_node[1]
    objective_lon = objective_node[0]

    h = EPSILON * manhattan_dist(current_lon, current_lat, objective_lon, objective_lat)
    NODES_EXPANDED += 1
    return h

def build_graph(detection_map: np.array, tolerance: np.float32) -> nx.DiGraph:
    """ Builds an adjacency graph (not an adjacency matrix) from the detection map """
    # The only possible connections from a point in space (now a node in the graph) are:
    #   -> Go up
    #   -> Go down
    #   -> Go left
    #   -> Go right
    # Not every point has always 4 possible neighbors
    graph = nx.DiGraph()
    height, width = detection_map.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for x in range(width):
        for y in range(height):
            graph.add_node((x, y))
    
    for x in range (width):
        for y in range(height):
            if detection_map[y, x] > tolerance:
                continue

            for dirX, dirY in directions:
                edgeX, edgeY = x + dirX, y + dirY
                if 0 <= edgeX < width and 0 <= edgeY < height and detection_map[edgeX, edgeY] <= tolerance:
                    graph.add_edge((x, y), (edgeX, edgeY), weight=detection_map[edgeX, edgeY])
    return graph
            
    


def discretize_coords(high_level_plan: np.array, boundaries: Boundaries, map_width: np.int32, map_height: np.int32) -> np.array:
    """ Converts coordiantes from (lat, lon) into (x, y) """
    max_lat = boundaries.max_lat
    min_lat = boundaries.min_lat
    max_lon = boundaries.max_lon
    min_lon = boundaries.min_lon

    step_lat = (max_lat - min_lat) / (map_height - 1)
    step_lon = (max_lon - min_lon) / (map_width - 1)

    grid_indices = np.empty((len(high_level_plan), 2), dtype=np.int32)

    for i, (lat, lon) in enumerate(high_level_plan):
        new_y = int(round((lat - min_lat) / step_lat))
        new_x = int(round((lon - min_lon) / step_lon))

        #insure clamping
        new_y = max(0, new_y)
        new_y = min(map_height - 1, new_y)
        new_x = max(0, new_x)
        new_x = min(map_width - 1, new_x)

        grid_indices[i] = (int(new_x), int(new_y))
    return grid_indices

def path_finding(G: nx.DiGraph,
                 heuristic_function,
                 locations: np.array, 
                 initial_location_index: np.int32, 
                 boundaries: Boundaries,
                 map_width: np.int32,
                 map_height: np.int32) -> tuple:
    """ Implementation of the main searching / path finding algorithm """
    path = []
    pois = tuple(map(tuple, discretize_coords(locations, boundaries, map_width, map_height)))
    
    curr_location = pois[initial_location_index]
    for poi in pois:
        if curr_location == poi:
            continue
        try:
            path_segment = nx.astar_path(G, curr_location, poi, heuristic_function, weight="weight")
            path.append(path_segment)
            curr_location = poi
        except nx.NetworkXNoPath:
            print(f"No path to poi from current location")
            continue
    return (path, NODES_EXPANDED)

def compute_path_cost(G: nx.DiGraph, solution_plan: list) -> np.float32:
    """ Computes the total cost of the whole planning solution """
    total_cost = 0.0
    
    for segment in solution_plan:
        for i in range(len(segment) - 1):
            total_cost += G[segment[i]][segment[i+1]]["weight"]
    
    return total_cost

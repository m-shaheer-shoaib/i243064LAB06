import math
import heapq

locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6)
}

hospital_graph = {
    "Pharmacy": {"Main_Corridor": 2.2, "Patient_Wing": 4.1},
    "Main_Corridor": {"Nursing_Station": 2.2},
    "Patient_Wing": {"Laboratory": 5.0},
    "Nursing_Station": {"Laboratory": 3.2, "Emergency_Ward": 6.0},
    "Laboratory": {"Emergency_Ward": 3.2},
    "Emergency_Ward": {}
}

def heuristic(current, goal):
    return math.dist(locations[current], locations[goal])

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def gbfs(start, goal):
    if start == goal:
        return [start], 0.0

    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), start))
    visited = set()
    came_from = {start: None}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current in visited:
            continue

        visited.add(current)

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            cost = sum(hospital_graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
            return path, cost

        for neighbor in hospital_graph[current]:
            if neighbor not in visited and neighbor not in came_from:
                came_from[neighbor] = current
                heapq.heappush(frontier, (heuristic(neighbor, goal), neighbor))

    return None, math.inf

def a_star(start, goal):
    if start == goal:
        return [start], 0.0

    frontier = []
    g_cost = {start: 0.0}
    came_from = {}
    heapq.heappush(frontier, (heuristic(start, goal), 0.0, start))

    while frontier:
        _, popped_g, current = heapq.heappop(frontier)

        if popped_g != g_cost.get(current, math.inf):
            continue

        if current == goal:
            return reconstruct_path(came_from, current), g_cost[current]

        for neighbor, edge_cost in hospital_graph[current].items():
            tentative_g = g_cost[current] + edge_cost

            if tentative_g < g_cost.get(neighbor, math.inf):
                g_cost[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(frontier, (f, tentative_g, neighbor))

    return None, math.inf

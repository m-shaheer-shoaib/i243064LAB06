import streamlit as st
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

# Graph, Use Case: Emergency Supply Robot
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

# Heuristic
def heuristic(current, goal):
    return math.dist(locations[current], locations[goal])

# Path reconstruction
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# GBFS
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

# A*
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

##########################################
# Streamlit GUI Code
# Set Page Config
st.set_page_config(page_title="Informed Search Visualizer", layout="wide")
st.title("Informed Search Visualizer")
st.markdown("Select an initial node, goal node, and informed search algorithm. The app runs the selected search on the hospital graph and highlights the returned solution path.")

# define the nodes and their coordinates
nodes = list(hospital_graph.keys())

# create a selectbox for the user to choose the start and goal nodes
start = st.selectbox(
    "Select Initial Node",
    nodes,
    index=nodes.index("Pharmacy")
)

goal = st.selectbox(
    "Select Goal Node",
    nodes,
    index=nodes.index("Emergency_Ward")
)

# create a selectbox for the user to choose the search algorithm
algorithm = st.selectbox("Select Search Algorithm", ["GBFS", "A*"])

if st.button("Run Search"):
    if algorithm == "GBFS":
        # run the GBFS algorithm with the selected start and goal nodes
        path, cost = gbfs(start, goal)
    else:
        # run the A* algorithm with the selected start and goal nodes
        path, cost = a_star(start, goal)

    if path is None:
        # display a error message indicating that no path was found
        st.error(f"No directed path exists from {start} to {goal}.")
    else:
        # Display result
        st.subheader("Search Result")
        st.write(f"Algorithm: {algorithm}")
        st.write(f"Solution Path: {' → '.join(path)}")
        st.write(f"Total Path Cost: {cost:.2f}")

        # Visualize NetworkX graph
        G = nx.DiGraph()
        for node, neighbors in hospital_graph.items():
            for neighbor, weight in neighbors.items():
                G.add_edge(node, neighbor, weight=weight)

        G.add_nodes_from(hospital_graph.keys())
        pos = locations
        fig, ax = plt.subplots(figsize=(10, 6))

        node_colors = [
            "green" if node == start else
            "red" if node == goal else
            "lightgray"
            for node in G.nodes()
        ]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2200, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=20, width=1.5, ax=ax)

        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, ax=ax)

        solution_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=solution_edges,
            edge_color="red",
            width=4,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=22,
            ax=ax
        )

        ax.set_title(f"{algorithm} Solution Path")
        ax.axis("off")
        st.pyplot(fig)

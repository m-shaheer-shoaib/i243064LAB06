import streamlit as st
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

# import the necessary functions and variables from searchAlgos.py
from searchAlgos import (
    hospital_graph,
    locations,
    gbfs,
    a_star
)

# Streamlit GUI
#*******************#
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
        st.write(f"**Algorithm:** {algorithm}")
        st.write(f"**Solution Path:** {' → '.join(path)}")
        st.write(f"**Total Path Cost:** {cost:.2f}")

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

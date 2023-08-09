import networkx as nx
import matplotlib.pyplot as plt
from sympy import symbols, sympify

x = symbols('x')
expr = sympify('4*exp(x) + 6*x**2')

G = nx.DiGraph()


def build_dag(expr, parent_name=None, graph=None):
    name = str(expr)
    graph.add_node(name, label=str(expr))

    if parent_name is not None:
        graph.add_edge(parent_name, name)

    for arg in expr.args:
        build_dag(arg, name, graph)


build_dag(expr, graph=G)

# 可视化
pos = nx.spring_layout(G)
labels = {v: G.nodes[v]['label'] for v in G.nodes()}
nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color='skyblue', font_size=10)
plt.show()
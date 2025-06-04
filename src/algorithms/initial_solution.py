"""Script for determining an initial solution using the LP relaxation of an ILP."""

import gurobipy as gp

from gurobipy import GRB
from src.utils.graph import Graph

class InitialSolutionGenerator:
    """
    Generates an initial solution for the maximum gamma-quasi-clique problem
    by solving the LP relaxation of the F3 formulation and rounding the k highest x values.
    """
    def __init__(self, gamma, k):
        """
        Initialize the generator with gamma parameter and target size k.
        
        Args:
            gamma (float): Quasi-clique density parameter (0 < gamma <= 1)
            k (int): Target size of the quasi-clique
        """
        self.gamma = gamma
        self.k = k
        
    def run(self, graph):
        """
        Generate initial solution by solving LP relaxation and rounding.
        
        Args:
            graph (Graph): Graph object to find quasi-clique in
            
        Returns:
            set: Set of node IDs forming the initial solution
        """
        if not isinstance(graph, Graph):
            raise ValueError("Input must be a Graph object")
            
        if self.k <= 0 or self.k > graph.num_vertices:
            raise ValueError(f"k must be between 1 and {graph.num_vertices}")
            
        if not (0 < self.gamma <= 1):
            raise ValueError("gamma must be between 0 and 1")
            
        # Get sorted vertices for consistent indexing
        vertices = sorted(graph.vertices)
        n = len(vertices)
        
        if n == 0:
            return set()

        model = gp.Model("QuasiClique_LP_Relaxation")
        model.setParam('OutputFlag', 0)
        
        x = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
        
        y = {}
        for i in range(n):
            for j in range(i+1, n):
                vertex_i = vertices[i]
                vertex_j = vertices[j]
                if graph.has_edge(vertex_i, vertex_j):
                    y[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"y_{i}_{j}")
    
        z = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"z_{self.k}")

        model.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
        edge_sum = gp.quicksum(y[i, j] for (i, j) in y.keys())
        required_edges = self.gamma * self.k * (self.k - 1) / 2
        model.addConstr(edge_sum >= required_edges * z, name="edge_density")
        
        for (i, j) in y.keys():
            model.addConstr(y[i, j] <= x[i], name=f"y_leq_xi_{i}_{j}")
            model.addConstr(y[i, j] <= x[j], name=f"y_leq_xj_{i}_{j}")
        
        model.addConstr(gp.quicksum(x[i] for i in range(n)) == self.k * z, name="size_constraint")
        model.addConstr(z == 1, name="z_fixed")
        
        try:
            model.optimize()
            
            if model.status != GRB.OPTIMAL:
                print(f"Warning: Model status is {model.status}, not optimal")
                if model.status == GRB.INFEASIBLE:
                    print("LP relaxation is infeasible - no solution exists")
                    return set()
                    
        except Exception as e:
            print(f"Error solving LP: {e}")
            return set()
        
        x_values = [(i, x[i].X) for i in range(n)]
        x_values.sort(key=lambda item: item[1], reverse=True)
        
        selected_indices = [i for i, _ in x_values[:self.k]]
        selected_vertices = {vertices[i] for i in selected_indices}
        
        if len(selected_vertices) != self.k:
            print(f"Warning: Expected {self.k} vertices but got {len(selected_vertices)}")
            
        return selected_vertices

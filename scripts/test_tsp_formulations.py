from sampco.co.tsp import *
from scipy.optimize import milp, LinearConstraint


import numpy as np
import matplotlib.pyplot as plt
import time

tsp = random_euclidean_tsp(15)

# Dantzig, Fulkerson and Johnson model
c, A, b,sense = tsp.ilp_formulation()
print("DFJ formulation")
print("\tc shape                  : ", c.shape)
print("\tA shape                  : ", A.shape)
print("\tb shape                  : ", b.shape)
print("\tNumber of variables      : ", c.shape[0])
print("\tNumber of constraints    : ", A.shape[0])
print("\tEquality constraints     : ", np.sum(sense == "="))
print("\tLess than constraints    : ", np.sum(sense == "<="))
print("\tGreater than constraints : ", np.sum(sense == ">="))
# solve the problem with MILP solver
lb, ub = np.full_like(b, -np.inf), np.full_like(b, np.inf)
lb[(sense == "=") | (sense == ">=")] = b[(sense == "=") | (sense == ">=")]
ub[(sense == "=") | (sense == "<=")] = b[(sense == "=") | (sense == "<=")]
t0 = time.time()
sol = milp(c, constraints=LinearConstraint(A, lb=lb, ub=ub), integrality=np.ones_like(c))
t0 = time.time() - t0
print("\tSolve time               : ", t0)
print("\tSolution                 : ", sol.x)
el = flow_to_edges(sol.x, tsp)
print("\tEdges                    : ", el)
route = ETSPSolution(edge_list_to_route(el))
print("\tRoute                    : ", route.sequence)
print("\tSolution cost            : ", np.dot(c, sol.x))
# Plot solution
fig,ax = plt.subplots(1,1)
plot_solution(route, tsp, ax=ax)
ax.set_title("TSP DFJ")
plt.savefig("tsp_dfj_sol.png")
plt.close(fig)

print("\n")

# Miller, Tucker and Zemlin model
c, A, b, sense = tsp.ilp_formulation(formulation="mtz")
print("MTZ formulation")
print("\tc shape                  : ", c.shape)
print("\tA shape                  : ", A.shape)
print("\tb shape                  : ", b.shape)
print("\tNumber of variables      : ", c.shape[0])
print("\tNumber of constraints    : ", A.shape[0])
print("\tEquality constraints     : ", np.sum(sense == "="))
print("\tLess than constraints    : ", np.sum(sense == "<="))
print("\tGreater than constraints : ", np.sum(sense == ">="))
# solve the problem with MILP solver
lb, ub = np.full_like(b, -np.inf), np.full_like(b, np.inf)
lb[(sense == "=") | (sense == ">=")] = b[(sense == "=") | (sense == ">=")]
ub[(sense == "=") | (sense == "<=")] = b[(sense == "=") | (sense == "<=")]
t0 = time.time()
sol = milp(c, constraints=LinearConstraint(A, lb=lb, ub=ub), integrality=np.ones_like(c))
t0 = time.time() - t0
print("\tSolve time               : ", t0)
print("\tSolution                 : ", sol.x)
el = flow_to_edges(sol.x[:tsp.number_of_edges()], tsp)
print("\tEdges                    : ", el)
route = ETSPSolution(edge_list_to_route(el))
print("\tRoute                    : ", route.sequence)
print("\tSolution cost            : ", np.dot(c, sol.x))
# Plot solution
fig,ax = plt.subplots(1,1)
plot_solution(route, tsp, ax=ax)
ax.set_title("TSP MTZ")
plt.savefig("tsp_mtz_sol.png")
plt.close(fig)


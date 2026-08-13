import gurobipy as gp
from gurobipy import GRB

# Create a small test model
model = gp.Model("test_gurobi")

x = model.addVar(lb=0, name="x")
y = model.addVar(lb=0, name="y")

# Minimize x + y
model.setObjective(x + y, GRB.MINIMIZE)

# Constraint: x + 2y >= 1
model.addConstr(x + 2 * y >= 1, name="c1")

model.optimize()

if model.status == GRB.OPTIMAL:
    print("Gurobi works.")
    print("Objective value:", model.objVal)
    print("x =", x.X)
    print("y =", y.X)
else:
    print("Gurobi did not solve successfully.")
    print("Status code:", model.status)
import sys
import numpy as np
from tsp_solver.greedy import solve_tsp

def total_distance(order, dist_matrix):
    return sum(dist_matrix[order[i], order[i+1]] for i in range(len(order) - 1))

def local_optimization(order, dist_matrix, iterations=1000):
    import numpy as np
    for _ in range(iterations):
        i, j = np.random.choice(len(order), 2, replace=False)
        new_order = order.copy()
        new_order[i], new_order[j] = new_order[j], new_order[i]
        if total_distance(new_order, dist_matrix) < total_distance(order, dist_matrix):
            order = new_order
    return order

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python reconstructOrder.py similarity_matrix.csv output_order.txt")
        sys.exit(1)
    dist_file = sys.argv[1]
    out_file = sys.argv[2]

    dist_matrix = np.loadtxt(dist_file, delimiter=",")
    initial_order = solve_tsp(dist_matrix)
    optimized_order = local_optimization(initial_order, dist_matrix)
    np.savetxt(out_file, optimized_order, fmt='%d')
    print(f"Reconstructed frame order saved to {out_file}")

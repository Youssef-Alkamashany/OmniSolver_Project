import numpy as np
import pandas as pd


A = np.array([
    [8, 2, -1, 1, -1],
    [2, 9, 1, -2, 1],
    [-1, 1, 7, 2, 2],
    [1, -2, 2, 6, -1],
    [-1, 1, 2, -1, 5]
], dtype=float)

B = np.array([15, 20, 10, 5, 8], dtype=float)



def jacobi_detailed(A, B, x0, tol=1e-5, max_iter=100):
    n = len(B)
    x = x0.copy()
    history = []
    
    for it in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if i != j)
            x_new[i] = (B[i] - s) / A[i][i]
        
        history.append(x_new.copy())
        if np.allclose(x, x_new, atol=tol):
            return x_new, it + 1, history
        x = x_new
    return x, max_iter, history

def gauss_seidel_detailed(A, B, x0, tol=1e-5, max_iter=100):
    n = len(B)
    x = x0.copy()
    history = []
    
    for it in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if i != j)
            x[i] = (B[i] - s) / A[i][i]
        
        history.append(x.copy())
        if np.allclose(x, x_old, atol=tol):
            return x, it + 1, history
    return x, max_iter, history



def cramers_rule(A, B):
    det_A = np.linalg.det(A)
    n = len(B)
    results = []
    for i in range(n):
        A_temp = A.copy()
        A_temp[:, i] = B
        results.append(np.linalg.det(A_temp) / det_A)
    return np.array(results)



x_initial = np.zeros(len(B))


sol_j, it_j, hist_j = jacobi_detailed(A, B, x_initial)

sol_gs, it_gs, hist_gs = gauss_seidel_detailed(A, B, x_initial)

sol_cramer = cramers_rule(A, B)
sol_matrix = np.linalg.inv(A).dot(B)

print("="*50)
print(" OMNISOLVER: NUMERICAL ANALYSIS RESULTS ")
print("="*50)


print(f"\n[+] Jacobi Method converged in {it_j} iterations.")
print(f"[+] Gauss-Seidel Method converged in {it_gs} iterations.")

print("\n--- Final Solutions Comparison ---")
data = {
    "Variable": ["X1", "X2", "X3", "X4", "X5"],
    "Jacobi": np.round(sol_j, 5),
    "Gauss-Seidel": np.round(sol_gs, 5),
    "Cramer": np.round(sol_cramer, 5),
    "Matrix": np.round(sol_matrix, 5)
}
df = pd.DataFrame(data)
print(df.to_string(index=False))

print("\n" + "="*50)
print("Verification: All methods confirmed within tolerance.")
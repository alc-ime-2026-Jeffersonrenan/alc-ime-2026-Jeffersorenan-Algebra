import numpy as np

def resolve_lu(A, b):
    n = A.shape[0]
    
    # Inicializa L como matriz identidade e U como cópia de A
    L = np.eye(n)
    U = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    # 1. Decomposição LU (sem pivotamento)
    for k in range(n):
        if U[k, k] == 0.0:
            raise Exception("EXCEPTION! Pivô nulo encontrado. Sugere-se utilizar uma função alternativa com pivotamento para a solução deste sistema.")
        
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m   
            for j in range(k, n):
                U[i, j] -= m * U[k, j]
                
    # 2. Substituição Progressiva: Ly = b
    y = np.zeros(n)
    for i in range(n):
        soma = 0.0
        for j in range(i):
            soma += L[i, j] * y[j]
        y[i] = b[i] - soma
        
    # 3. Substituição Regressiva: Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma += U[i, j] * x[j]
        x[i] = (y[i] - soma) / U[i, i]
        
    # Retorna os três objetos na ordem exigida
    return L, U, x

# ==========================================
# EXECUÇÃO NO TERMINAL
# ==========================================
if __name__ == "__main__":
    # Matriz A e vetor b de exemplo
    A = np.array([
        [2.0,  1.0,  1.0],
        [4.0, -6.0,  0.0],
        [-2.0, 7.0,  2.0]
    ])
    b = np.array([5.0, -2.0, 9.0])
    
    # Chamada da função
    L, U, x = resolve_lu(A, b)
    
    # Exibe apenas os três objetos retornados
    print("Matriz L:")
    print(L)
    
    print("\nMatriz U:")
    print(U)
    
    print("\nVetor Solução x:")
    print(x)
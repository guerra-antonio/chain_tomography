# Importamos librerias necesarias
import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm

# ------------------------------
# Definición de MUBs (4D)
# ------------------------------
c = np.complex64
MUB1 = np.array(np.eye(4), dtype=c)

MUB2 = np.array([
    [1/2,  1/2,  1/2,  1/2],
    [1/2, -1/2,  1/2, -1/2],
    [1/2,  1/2, -1/2, -1/2],
    [1/2, -1/2, -1/2,  1/2]
], dtype=c)

MUB3 = np.array([
    [1j/2,  1j/2,  1j/2,  1j/2],
    [-1j/2, 1j/2, -1j/2,  1j/2],
    [1/2,   1/2, -1/2,  -1/2],
    [1/2,  -1/2, -1/2,   1/2]
], dtype=c)

MUB4 = np.array([
    [-1/2,  -1/2,  -1/2, -1/2],
    [-1j/2,  1j/2, -1j/2, 1j/2],
    [-1j/2, -1j/2,  1j/2, 1j/2],
    [ 1/2,  -1/2,  -1/2, 1/2]
], dtype=c)

MUB5 = np.array([
    [1j/2,  1j/2,  1j/2,  1j/2],
    [1/2,  -1/2,   1/2,  -1/2],
    [-1j/2, -1j/2,  1j/2, 1j/2],
    [1/2,  -1/2,  -1/2,  1/2]
], dtype=c)

# Lookup por ENTERO (evitamos cast a str)
MUBS = {1: MUB1, 2: MUB2, 3: MUB3, 4: MUB4, 5: MUB5}


# ------------------------------
# Leemos los datos
# ------------------------------
def extract_data(df, n: int):
    """Toma 4 filas (un bloque), devuelve projectores y probabilidades."""
    df_tomo = df.iloc[4*n:4*(n+1), :]

    counts = df_tomo.iloc[:, :4].to_numpy(dtype=float)
    # normaliza por fila y aplana (16 probs)
    probs = (counts / counts.sum(axis=1, keepdims=True)).reshape(-1)

    mubs_label = df_tomo.iloc[:, 4].astype(int).to_numpy()

    # 4 projectores por fila (total 16)
    projectors = []
    for label in mubs_label:
        U = MUBS[label]
        for col in range(U.shape[1]):
            v = U[:, col].reshape(4, 1)
            P = v @ v.conj().T
            projectors.append(P)
    
    # Añadimos la MUB0
    U = MUBS[1]
    for col in range(U.shape[1]):
        v = U[:, col].reshape(4, 1)
        P = v @ v.conj().T
        projectors.append(P)
    
    probs = np.concatenate([probs, np.full(4, 0.25)])
    projectors = np.stack(projectors, axis=0)  # (16,4,4)

    return projectors, probs


# ------------------------------
# Tomografía
# ------------------------------
def tomography(data):
    rho_list = []

    n_tomo = data.shape[0] // 4

    for t in range(n_tomo):
        Proj, p = extract_data(df = data, n = t)

        rho = cp.Variable((4, 4), hermitian=True)
        constraints = [rho >> 0, cp.trace(rho) == 1]

        # Sum of squares: más idiomático y estable
        preds = [cp.real(cp.trace(rho @ Proj[i])) for i in range(len(p))]
        objective = cp.Minimize(cp.sum_squares(cp.hstack(preds) - p))

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-6, max_iters=5000)

        if rho.value is None:
            raise RuntimeError(f"El solver falló. Estado: {prob.status}")

        # Limpieza numérica
        rho_est = (rho.value + rho.value.conj().T) / 2
        P = float(np.real(np.trace(rho_est @ rho_est)))

        rho_list.append(rho_est)

    return rho_list

# Fidelidad
def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Calcula la fidelidad cuántica entre dos matrices densidad mixtas.
    
    Parámetros
    ----------
    rho : np.ndarray
        Matriz densidad (Hermítica, traza 1).
    sigma : np.ndarray
        Matriz densidad (Hermítica, traza 1).

    Retorna
    -------
    float
        Fidelidad entre rho y sigma, en [0, 1].
    """
    # Asegurar tipo complejo
    rho = np.array(rho, dtype=np.complex128)
    sigma = np.array(sigma, dtype=np.complex128)

    # Raíz cuadrada de rho
    sqrt_rho = sqrtm(rho)

    # Producto intermedio
    inner = sqrt_rho @ sigma @ sqrt_rho

    # Raíz cuadrada del producto
    sqrt_inner = sqrtm(inner)

    # Fidelidad (traza del operador raíz al cuadrado)
    F = np.real(np.trace(sqrt_inner)) ** 2

    # Limpieza numérica
    if np.isnan(F) or np.isinf(F):
        return 0.0
    return float(np.clip(F, 0.0, 1.0))
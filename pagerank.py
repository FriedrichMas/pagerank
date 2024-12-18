import numpy as np

# 1. Matrice C : Représente les liens
C = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Page 1 ne reçoit aucun lien
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Page 2 reçoit des liens de 1 et 4
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Page 3 reçoit des liens de 2
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Page 4 reçoit des liens de 2 et 3
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # page 5 reçoit des liens de 1 et 6
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # page 6 reçoit des liens de 6
    [0, 0, 0, 1, 0, 1, 0, 0, 1, 0],  # page 7 reçoit des liens de 4, 6 et 9
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # page 8 reçoit des liens de 7 et 9
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # page 9 reçoit des liens de 8
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # page 10 reçoit des liens de 3
])

# 2. Matrice Q : Normalisation des colonnes
N = C.shape[0]  # Nombre de pages
Q = np.zeros_like(C, dtype=float)

for j in range(N):
    col_sum = np.sum(C[:, j])
    if col_sum > 0:
        Q[:, j] = C[:, j] / col_sum

# 3. Matrice P : Ajout des colonnes nulles
e = np.ones(N)
d = (np.sum(C, axis=0) == 0).astype(float)  # Colonnes nulles
P = Q + (1 / N) * np.outer(e, d)

# 4. Matrice A : Ajout du paramètre alpha
alpha = 0.85
A = alpha * P + (1 - alpha) / N * np.outer(e, e)

print("Matrice C:\n", C)
print("Matrice Q:\n", Q)
print("Matrice P:\n", P)
print("Matrice A:\n", A)

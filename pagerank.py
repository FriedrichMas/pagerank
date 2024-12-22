import numpy as np
import matplotlib.pyplot as plt

#PARTIE I

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
N = C.shape[0]  # Renvoie la taille de la première dimension (ligne) de C, donc nombre de pages
Q = np.zeros_like(C, dtype=float) #Crée une matrice de zéros ayant le même adn que C

for j in range(N):
    somme_colonnes = np.sum(C[:, j])
    if somme_colonnes > 0:
        Q[:, j] = C[:, j] / somme_colonnes

# 3. Matrice P : Ajout des colonnes nulles
e = np.ones(N)
d = (np.sum(C, axis=0) == 0).astype(float)  # Colonnes nulles
P = Q + (1 / N) * np.outer(e, d)

# 4. Matrice A : Ajout du paramètre alpha (compris entre 0 et 1)
alpha = 0.85
A = alpha * P + (1 - alpha) / N * np.outer(e, e)

print("Matrice C:\n", C)
print("Matrice Q:\n", Q)
print("Matrice P:\n", P)
print("Matrice A:\n", A)


#PARTIE II

# 1. Définir les paramètres
b = (1 - alpha) / N * e
I = np.eye(N)
A_2 = I - alpha * P  # Matrice du système
w_values = np.linspace(0.1, 1.9, 100)  # Paramètres de relaxation
iterations = []

# 2. Méthode de relaxation
for w in w_values:
    r = np.zeros(N)  # Initialisation
    max_iter = 1000
    tolerance = 1e-6
    num_iter = 0
    
    for k in range(max_iter):
        #r_new = r + w * (b - A_2 @ r)  # Itération de relaxation
        M = (1/w)*np.diag(A_2) + np.tril(A_2)
        Nn = ((1-w)/w)*np.diag(A_2) + np.triu(A_2)
        M_inv = np.linalg.inv(M)
        Lw = M_inv @ Nn
        r_new = Lw @ r + M_inv @ b
        if np.linalg.norm(r_new - r, 1) < tolerance:
            break
        r = r_new
        num_iter += 1
    
    iterations.append(num_iter)
print("le vecteur r est :\n", r_new)

# 3. Tracer la courbe
plt.plot(w_values, iterations)
plt.xlabel("Paramètre de relaxation (w)")
plt.ylabel("Nombre d'itérations")
plt.title("Convergence en fonction de w")
plt.grid()
plt.show()


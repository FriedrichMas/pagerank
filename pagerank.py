import numpy as np
import matplotlib.pyplot as plt

# PARTIE I : Construction des matrices

# 1. Matrice C : Représentant les liens
C = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Page 1 ne reçoit aucun lien
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Page 2 reçoit des liens de 1 et 4
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Page 3 reçoit des liens de 2
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Page 4 reçoit des liens de 2 et 3
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # page 5 reçoit des liens de 1 et 6
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # page 6 reçoit des liens de 5
    [0, 0, 0, 1, 0, 1, 0, 0, 1, 0],  # page 7 reçoit des liens de 4, 6 et 9
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # page 8 reçoit des liens de 7 et 9
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # page 9 reçoit des liens de 8
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # page 10 reçoit des liens de 3
])

# 2. Matrice Q : Normalisation des colonnes
N = C.shape[0] #Renvoie la taill de la première dimension(ligne) de C, donc le nombre des pages. 
Q = np.zeros_like(C, dtype=float) #Crée une matrice de zéros ayant le même adn que C 

for j in range(N):
    somme_colonnes = np.sum(C[:, j])
    if somme_colonnes > 0:
        Q[:, j] = C[:, j] / somme_colonnes

# 3. Matrice P : Ajout des colonnes nulles
e = np.ones(N)
d = (np.sum(C, axis=0) == 0).astype(float)  #Colonnes nulles
P = Q + (1 / N) * np.outer(e, d)

# 4. Matrice A : Ajout du paramètre alpha (compris entre 0 et 1)
alpha = 0.85
A = alpha * P + (1 - alpha) / N * np.outer(e, e)

# Affichages
print("Matrice C:\n", C)
print("Matrice Q:\n", Q)
print("Matrice P:\n", P)
print("Matrice A:\n", A)

print("\n" + " " * 50 + "-" * 100 + "\n")

# METHODE DE LA PUISSANCE ITEREE
rp = np.ones(N) / N  # Vecteur initial uniformément reparti
tolerance = 1e-6
max_iter = 1000

for i in range(max_iter):
    r_new = A @ rp   #itérations
    r_new /= np.linalg.norm(r_new, 2)
    if np.linalg.norm(r_new - rp, 2) < tolerance:
        break
    rp = r_new

print("Vecteur propre r (classement des pages) :\n", rp)

# Classement des pages
print("\nClassement des pages obtenu par la méthode de puissance itérée :")
indices_triees = np.argsort(rp)[::-1] #scores triés en ordre décroissant
classement = [(f"page{i+1}", rp[i]) for i in indices_triees] #création de l'affichage des pages et scores triés 
for rang, (page, score) in enumerate(classement, start=1):
    print(f"{rang}. {page} : {score}") #affichage des resultats

print("\n" + " " * 50 + "-" * 100 + "\n")

# PARTIE II : METHODE DE RELAXATION
b = (1 - alpha) / N * e
I = np.eye(N)
A_2 = I - alpha * P


# Fonction de relaxation
def relax(A_2, b, w, max_iter, tol):
    N = A_2.shape[0]
    r = np.zeros(N)
    for i in range(max_iter):
        try:
            M = (1 / w) * np.diag(np.diag(A_2)) + np.tril(A_2, k=-1)
            Nn = ((1 - w) / w) * np.diag(np.diag(A_2)) - np.triu(A_2, k=1)
            M_inv = np.linalg.inv(M)
            Lw = M_inv @ Nn #Matrice d'itération
            r_new = Lw @ r + M_inv @ b #formule d'itération de relaxation

            # Vérification de stabilité
            if not np.all(np.isfinite(r_new)):
                print(f"NaN ou Inf détecté à l'itération {i} avec w = {w}")
                return r, i + 1

            if np.linalg.norm(r_new - r, ord=1) < tol:
                return r_new, i + 1

            r = r_new
        except Exception as e:
            print(f"Erreur à l'itération {i} avec w = {w}: {e}")
            break
    return r, max_iter

# Paramètres pour relaxation
w_values = np.linspace(0.1, 1.9, 100)
iterations = []

for w in w_values:
    _, num_iter = relax(A_2, b, w, max_iter, tolerance)
    iterations.append(num_iter)


# Affichage des résultats

# Trouver le w correspondant au minimum global de la courbe
min_iterations = min(iterations)  # Nombre minimum d'itérations
optimal_w = w_values[np.argmin(iterations)]  # w correspondant

# Affichage des résultats
print(f"Le paramètre w optimal est : {optimal_w:.4f}")
print(f"Il donne le nombre minimal d'itérations, qui est : {min_iterations}")

# Optionnel : Marquer ce point sur le graphique
plt.plot(w_values, iterations, label="Nombre d'itérations")
plt.axvline(optimal_w, color="green", linestyle="--", label=f"w optimal = {optimal_w:.4f}")
plt.axvline(1.0, color="red", linestyle="--", label="Relaxation standard (w=1)")
plt.scatter([optimal_w], [min_iterations], color="green", zorder=5, label="Minimum global")
plt.xlabel("Paramètre de relaxation (w)")
plt.ylabel("Nombre d'itérations")
plt.title("Convergence de la méthode de relaxation")
plt.legend()
plt.grid()
plt.show()


plt.title("Convergence en fonction de w")
plt.grid()
plt.show()


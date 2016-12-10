import numpy as np


# Canberra-Distanz
def Canberra(p, q):
    can = 0
    for i in range(len(p)):
        can = can + abs(p[i] - q[i]) / (abs(p[i]) + abs(q[i]))
    return can


# Chebyshev-Distanz
def Chebyshev(p, q):
    return np.amax(abs(p - q))


# Kosinusähnlichkeit
def Kosinus(p, q):  # auf shape achten!!!
    return np.matmul(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))

a = np.array([2, 2, 3, 4])
b = np.array([1, 2, 3, 5])

# Was bringen diese Abstände?!
print(Canberra(a, b))
print(Kosinus(a, b))
print(Chebyshev(a, b))

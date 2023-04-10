import numpy as np
import matplotlib.pyplot as plt

Omat = np.array([[0, 1], [-1, 0]])

def line_through(A, B):
    m = B-A
    n = Omat@m
    return n, n.T@B

def foot_of_perp(n, c, P):
    m = Omat@n
    X = np.array([m, n])
    O = np.array([m.T@P, c])
    return np.linalg.solve(X, O)

AE = 8
AD = 12.8

theta = np.arcsin(AE/AD)

A = AD * np.array([np.cos(theta), np.sin(theta)])
B = np.array([A[0] + 16, 8])
C = np.array([16, 0])
D = np.array([ 0, 0])
E = np.array([A[0], 0])

n,c = line_through(A, D)
F = foot_of_perp(n, c, C)
    
def plt_pnt(A, label=''):
	plt.plot(A[0], A[1], 'o')
	if label != '': plt.text(A[0], A[1], label)
		
def plt_line(A, B, labelA='', labelB=''):
    plt.plot([A[0],B[0]], [A[1],B[1]], label=labelA+labelB)
    if labelA != '': plt_pnt(A, labelA)
    if labelB != '': plt_pnt(B, labelB)

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

print("F =", F)
print("CF =", np.linalg.norm(C-F))
print("Angle b/w CF, AD =", angle_between(C-F, A-D))

plt_line(A, B, 'A', 'B')
plt_line(B, C, '', 'C')
plt_line(C, D, '', 'D')
plt_line(D, A)
plt_line(A, E, '', 'E')
plt_line(C, F, '', 'F')

plt.grid(), plt.axis('square')
plt.show()

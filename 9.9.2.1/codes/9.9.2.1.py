import numpy as np
import matplotlib.pyplot as plt

A = np.array([np.sqrt(99.84), 8])
B = np.array([A[0] + 16, 8])
C = np.array([16, 0])
D = np.array([ 0, 0])
E = np.array([A[0], 0])

Omat = np.array([[0, 1], [-1, 0]])
X = np.vstack((C, A@Omat))
O = np.array([156, 0])

F = np.linalg.solve(X, O)
    
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

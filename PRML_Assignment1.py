import numpy as np
import matplotlib.pyplot as plt
		
def construct_triangle(BC, angle_B, AB_plus_AC):
    a = BC
    K = AB_plus_AC
    B = np.deg2rad(angle_B)

    X = np.array([ [1,  1], [K, 2*a*np.cos(B) - K] ])
    D = np.array([ K, a*a ])
    c = np.linalg.solve(X, D)[1]

    A = (c * np.cos(B), c * np.sin(B))
    B = (0, 0)
    C = (a, 0)

    plt_line(A, B, 'A', 'B')
    plt_line(B, C, 'B', 'C')
    plt_line(C, A, 'C', 'A')
    
def plt_pnt(A, label=''):
	plt.plot(A[0], A[1], 'o')
	if label != '':
		plt.text(A[0], A[1], label)
		
def plt_line(A, B, labelA='', labelB='', plt_plts = True):
	plt.plot([A[0],B[0]], [A[1],B[1]], label=labelA+labelB)
	if plt_plts:
		plt_pnt(A, labelA)
		plt_pnt(B, labelB)
		
construct_triangle(7, 75, 13)

plt.grid(), plt.axis('equal')
plt.show()
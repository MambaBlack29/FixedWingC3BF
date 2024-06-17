import sympy as sp 
from sympy import cos, sin, tan
from sympy.abc import t
import numpy as np

# for symbolic calculation of the gr fr grd and frd constants
# state and input
time_func = lambda x : sp.Function(x, real = True)(t)

n, e, d, phi, theta, psi, vt, = list(map(time_func, 'n e d phi theta psi vt'.split()))
At, P, Q, g, mu = sp.symbols('At P Q g mu', real = True)

# relevant variables for state
x = [n, e, d, phi, theta, psi, vt]
u = sp.Matrix([At, P, Q])
r = sp.Matrix([n, e, d])
v = sp.Matrix([vt*cos(theta)*cos(psi),
               vt*cos(theta)*sin(psi),
               -vt*sin(theta)])

# goals
r1, r2, r3 = list(map(time_func, 'r1 r2 r3'.split()))
rg = sp.Matrix([r1,r2,r3])
vg = rg.diff(t)
ag = vg.diff(t)

# state space
R = g*sin(phi)*cos(theta)/vt
R_dot = R.diff(t)

n_dot = vt*cos(psi)*cos(phi)
e_dot = vt*sin(psi)*cos(phi)
d_dot = -vt*sin(theta)
phi_dot = P + sin(phi)*tan(theta)*Q + cos(phi)*tan(theta)*R
theta_dot = cos(phi)*Q - sin(phi)*R
psi_dot = (sin(phi)*Q + cos(phi)*R)/cos(theta)
v_dot = At

x_dot = [n_dot, e_dot, d_dot, phi_dot, theta_dot, psi_dot, v_dot]

g_1 = sp.Matrix([[0, 1, sin(phi)*tan(theta)],
                 [0, 0, cos(phi)],
                 [0, 0, sin(phi)/cos(theta)]])
g_2 = sp.Matrix([[1,0,0]])
G = sp.Matrix(sp.BlockMatrix([[sp.zeros(3)],[g_1],[g_2]]))
print(G.shape)

# obstacle path
c1, c2, c3, c1_dot, c2_dot, c3_dot = list(map(time_func, 'c1 c2 c3 c1_dot c2_dot c3_dot'.split()))
c = [c1, c2, c3, c1_dot, c2_dot, c3_dot]
c_dot = [c1_dot, c2_dot, c3_dot, 0, 0, 0]

rad = sp.Symbol('rad', real = True)
obr = sp.Matrix([c1, c2, c3])
obv = sp.Matrix([c1_dot, c2_dot, c3_dot])

# rotation matrix from body to earth frame
Rot = sp.Matrix([[cos(psi)*cos(theta), cos(psi)*sin(phi)*sin(theta) - cos(phi)*sin(psi), sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta)],
                 [cos(theta)*sin(psi), cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta), cos(phi)*sin(psi)*sin(theta) - cos(psi)*sin(phi)],
                 [-sin(theta), cos(theta)*sin(phi), cos(phi)*cos(theta)]])
lx, ly, lz = sp.symbols('lx, ly, lz', real = True) # COM in body frame
l_com = sp.Matrix([lx, ly, lz])
print(l_com)

# dictionary to substitute d(whatever)/dt to actual values from the state space
diff_func = lambda y : y.diff(t)
diff_list = list(map(diff_func, x))
diff_c_list = list(map(diff_func, c))
sub_dict_c = dict({diff_c_list[i]:c_dot[i] for i in range(len(c))})
sub_dict_x = dict({diff_list[i]:x_dot[i] for i in range(len(x))})

# relative
prel = obr - r - Rot*l_com
vrel = sp.simplify(prel.diff(t).subs(sub_dict_c).subs(sub_dict_x))
print(vrel)
# c3bf with backstepping
h = sp.Matrix([sp.simplify(((prel.T*vrel)[0,0] + sp.simplify(vrel.norm()*(prel.norm()**2 - rad**2)**0.5)).subs(sub_dict_x))])
del_h = sp.simplify(h.jacobian(l_com))

print("del h / del l:")
sp.pprint(del_h)
# print("LgH is:")
# print(sp.simplify(del_h*G))

# v_dot = sp.simplify(vrel.diff(t).subs(sub_dict_c).subs(sub_dict_x))

# print("\nv_dot is:\n")
# print(v_dot)
# print("\nThe part with u:\n")
# v_dot_u = sp.simplify(v_dot.jacobian(u))
# print(v_dot_u)
# print("\nThe part without u:\n")
# v_dot_nu = sp.simplify(v_dot - v_dot_u*u)
# print(v_dot_nu)
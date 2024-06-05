import sympy as sp 
from sympy import cos, sin, tan
from sympy.abc import t
import numpy as np

# for symbolic calculation of the gr fr grd and frd constants
# state and input
time_func = lambda x : sp.Function(x, real = True)(t)

n, e, d, phi, theta, psi, vt, = list(map(time_func, 'n e d phi theta psi vt'.split()))
At, P, Q, g = sp.symbols('At P Q g', real = True)

# relevant variables for state
x = [n, e, d, phi, theta, psi, vt]
u = [At, P, Q]
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
vt_dot = At

x_dot = [n_dot, e_dot, d_dot, phi_dot, theta_dot, psi_dot, vt_dot]

# obstacle path
c1, c2, c3, c1_dot, c2_dot, c3_dot = list(map(time_func, 'c1 c2 c3 c1_dot c2_dot c3_dot'.split()))
c = [c1, c2, c3, c1_dot, c2_dot, c3_dot]
c_dot = [c1_dot, c2_dot, c3_dot, 0, 0, 0]

rad = sp.Symbol('rad', real = True)
obr = sp.Matrix([c1, c2, c3])
obv = sp.Matrix([c1_dot, c2_dot, c3_dot])

# dictionary to substitute d(whatever)/dt to actual values from the state space
diff_func = lambda y : y.diff(t)
diff_list = list(map(diff_func, x))
diff_c_list = list(map(diff_func, c))
sub_dict_c = dict({diff_c_list[i]:c_dot[i] for i in range(len(c))})
sub_dict_x = dict({diff_list[i]:x_dot[i] for i in range(len(x))})

# relative
prel = obr - r
vrel = obv - v

# c3bf alone
cos_phi = (prel.norm()**2 - rad**2)**0.5/(prel.norm())
h = sp.simplify((prel.T*vrel)[0,0] + prel.norm()*vrel.norm()*cos_phi)
h_dot = sp.simplify(h.diff(t).subs(sub_dict_c).subs(sub_dict_x))
sp.pprint(h_dot)

# v_dot = sp.simplify(v.diff(t).subs(sub_dict))

# print("\nv_dot is:\n")
# sp.pprint(v_dot)
# print("\nThe part with u:\n")
# v_dot_u = -v_dot.jacobian(u)
# sp.pprint(v_dot_u)

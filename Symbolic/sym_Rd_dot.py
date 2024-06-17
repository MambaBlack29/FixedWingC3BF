import sympy as sp 
from sympy import cos, sin, tan
from sympy.abc import t
import numpy as np

# for symbolic calculation of the gr fr grd and frd constants
# state and input
time_func = lambda x : sp.Function(x)(t)

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
v_dot = At

x_dot = [n_dot, e_dot, d_dot, phi_dot, theta_dot, psi_dot, v_dot]

# dictionary to substitute d(whatever)/dt to actual values from the state space
diff_func = lambda y : y.diff(t)
diff_list = list(map(diff_func, x))
sub_dict = dict({diff_list[i]:x_dot[i] for i in range(len(x))})

# velocity tracking
kr1, kr2, kr3, kv1, kv2, kv3 = sp.symbols('kr1 kr2 kr3 kv1 kv2 kv3', real = True)
Kr = sp.diag(kr1, kr2, kr3)
Kv = sp.diag(kv1, kv2, kv3)

vc = vg + Kr*(rg - r)
ac = vc.diff(t)
ad = ac + Kv*(vc - v)/2

Ma = sp.Matrix([[cos(theta)*cos(psi), -vt*(cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi)), vt*(sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi))],
                [cos(theta)*sin(psi), vt*(-cos(phi)*sin(theta)*sin(psi) + sin(phi)*cos(psi)), vt*(sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi))],
                [-sin(theta), -vt*cos(phi)*cos(theta), vt*sin(phi)*cos(theta)]])

Ma_inv = Ma.inv()
inter_input = (Ma_inv*ad)

Rd = inter_input[2]
Rd_dot = sp.Matrix([Rd.diff(t)])

Rd_dot = sp.simplify(Rd_dot.subs(sub_dict))
Rd_dot_u = Rd_dot.jacobian(u)

sp.pprint(sp.simplify(Rd_dot_u))
# R_dot = sp.simplify(R_dot.subs(sub_dict))

# gr = R_dot.diff(P)
# fr = sp.simplify(R_dot - gr*P)

# frd = sp.simplify(Rd_dot.subs(P,0).subs(sub_dict).subs(P,0))
# grd = (Rd_dot - frd).diff(P)

# print(gr)
# print()
# print(fr)
# print()
# print(grd)
# print()
# print(frd)
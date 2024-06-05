import numpy as np
from numpy import sin, cos, tan
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.interpolate as si

# vista model, taken from collision avoidance and geofencing for fixed wing aircraft with control barrier functions
g = 9.81 # gravity

# -----------RELEVANT FUNCTIONS-----------
# want rg, vg, ac = generate trajectory
def trajectory(waypoints, delta, t0, tf):
    # u = on a 0-1 scale where the points are located
    tck, u = si.splprep(waypoints.T, k=3, s=0)

    # how finely want to sample the spline, basically time parameter
    u_fine = np.arange(t0, tf, delta)

    # spline points and relevant derivatives in 3D
    rg = np.array(si.splev(u_fine, tck)).T 
    vg = np.array(si.splev(u_fine, tck, der=1)).T
    ag = np.array(si.splev(u_fine, tck, der=2)).T
    
    return np.concatenate((rg, vg, ag), axis = 1)

# waypoint generator based on r0 and v0 (a = 0)
def way_gen(r0, v0):
    way_points = []
    for i in range(101):
        next_point = r0 + i/100*v0
        way_points.append(next_point)
    return np.array(way_points)

# total state x = n e d phi theta psi vt
# x_dot function as the state model
def x_dot(t, x, traj_self, traj_ob, delta, con_params):
    # depackage
    [n, e, d, phi, theta, psi, vt] = x

    # system matrix (non linear) f(x)
    f_1 = np.array([vt*cos(theta)*cos(psi),
                  vt*cos(theta)*sin(psi),
                  -vt*sin(theta)])
    f_2 = g/vt * np.array([sin(phi)*cos(phi)*sin(theta),
                            -(sin(phi)**2)*cos(theta),
                            sin(phi)*cos(phi)])
    f_3 = np.array([0])
    F = np.concatenate((f_1, f_2, f_3))

    # control matrix (non linear) g(x)
    g_1 = np.array([[0, 1, sin(phi)*tan(theta)],
                    [0, 0, cos(phi)],
                    [0, 0, sin(phi)/cos(theta)]])
    g_2 = np.array([[1,0,0]])
    G = np.concatenate((np.zeros((3,3)), g_1, g_2))

    # next desired point based on trajectory and given time
    x_des = traj_self[int(t/delta) - 1]
    x_ob = traj_ob[int(t/delta) - 1]

    u = controller(x, x_des, x_ob, con_params)

    # x_dot = f(x) + g(x)*u control affine system
    return F + np.dot(G, u)

# system input u = At P Q (linear accel, roll rate, pitch rate)
# x_des through trajectory per time step = rg vg ag
def controller(x, x_des, x_ob, con_params):
    # depackage into relevant variables
    [n, e, d, phi, theta, psi, vt] = x
    r = np.array([n,e,d])
    rg = np.array(x_des[:3])
    vg = np.array(x_des[3:6])
    ag = np.array(x_des[6:])

    # matrices and constants
    Kr, Kv, mu, rad, mu_e, W = con_params
    l = 2/3*min(np.linalg.eig(Kv)[0]) # smallest eigenvalue

    # getting vc = vd (commanded vel = desired vel)
    er = rg - r
    vd = vg +  np.dot(Kr, er.T)

    # getting ad
    v = np.array([vt*cos(theta)*cos(psi),
                  vt*cos(theta)*sin(psi),
                  -vt*sin(theta)])
    ev = vd - v
    ac = ag + np.dot(Kr, (vg - v).T)
    ad = ac + np.dot(Kv, ev.T)/2

    # getting At Q Rd
    Ma = np.array([[cos(theta)*cos(psi), -vt*(cos(phi)*cos(psi)*sin(theta) + sin(phi)*sin(psi)), vt*(sin(phi)*cos(psi)*sin(theta) - cos(phi)*sin(psi))],
                   [cos(theta)*sin(psi), vt*(-cos(phi)*sin(psi)*sin(theta) + sin(phi)*cos(psi)), vt*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))],
                   [-sin(theta), -vt*cos(phi)*cos(theta), vt*sin(phi)*cos(theta)]])
    At, Q, Rd = np.dot(np.linalg.inv(Ma), ad.T)

    # getting P (values of the variables found symbolically)
    R = g*sin(phi)*cos(theta)/vt # yaw rate
    Mr = Ma[:, 2] # column 3

    v_dot = ad + (R - Rd)*Mr # actual acceleration after control action

    gr = g*cos(phi)*cos(theta)/vt

    fr = g*(-At + g*sin(theta))*sin(phi)*cos(theta)/vt**2

    grd = ((2*vt*cos(phi - theta) + 2*vt*cos(phi + theta))*(2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])/4 - (vt*sin(phi)*sin(psi) + vt*sin(theta)*cos(phi)*cos(psi))*(2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) + Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) - 2*ag[0]) + (vt*sin(phi)*cos(psi) - vt*sin(psi)*sin(theta)*cos(phi))*(2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) + Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) - 2*ag[1]))/(2*vt**2)

    frd = (-4*(-(2*Kr[0,0]*((Q*vt + g*cos(phi)*cos(theta))*sin(phi)*sin(psi)*cos(phi) + (-At*cos(phi)*cos(psi) + (Q*vt*tan(theta) + g*sin(theta)*cos(phi))*sin(phi)**2*cos(psi) + ag[0])*cos(theta)) + (Kv[0,0]*(-At*cos(psi)*cos(theta) - Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) + (Q*vt + g*cos(phi)*cos(theta))*sin(phi)*sin(psi) + (Q*vt*cos(phi) - g*sin(phi)**2*cos(theta))*sin(theta)*cos(psi) + ag[0]))*cos(theta))*(sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi)) + (2*Kr[1,1]*((Q*vt + g*cos(phi)*cos(theta))*sin(phi)*cos(phi)*cos(psi) + (At*sin(psi)*cos(phi) - (Q*vt*tan(theta) + g*sin(theta)*cos(phi))*sin(phi)**2*sin(psi) - ag[1])*cos(theta)) + (Kv[1,1]*(At*sin(psi)*cos(theta) + Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) + (Q*vt + g*cos(phi)*cos(theta))*sin(phi)*cos(psi) - (Q*vt*cos(phi) - g*sin(phi)**2*cos(theta))*sin(psi)*sin(theta) - ag[1]))*cos(theta))*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi)))*vt + (4*At*((sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) + Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) - 2*ag[1]) + (sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi))*(2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) + Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) - 2*ag[0])) - 4*At*(2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*sin(phi)*cos(theta) + g*(cos(phi - 2*theta) - cos(phi + 2*theta))*(2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2]) + 4*g*(2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) + Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) - 2*ag[0])*sin(phi)*cos(psi)*cos(theta)**2 + 4*g*(2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) + Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) - 2*ag[1])*sin(phi)*sin(psi)*cos(theta)**2 + 4*(2*Kr[2,2]*(At*sin(theta) + (Q*vt*cos(phi) - g*sin(phi)**2*cos(theta))*cos(theta) + ag[2]) + Kv[2,2]*(At*sin(theta) + Kr[2,2]*(vt*sin(theta) + vg[2]) + (Q*vt*cos(phi) - g*sin(phi)**2*cos(theta))*cos(theta) + ag[2]))*vt*sin(phi)*cos(theta))*cos(theta))/(8*vt**2*cos(theta))

    a1 = -np.dot(ev, np.dot(Kv, ev.T))/2
    a2 = np.dot(ev, Mr.T)*(Rd - R)
    a3 = (Rd - R)*(frd - fr)/mu
    a4 = l*(np.dot(ev, ev) + ((Rd - R)**2)/mu)/2
    ap = a1 + a2 + a3 + a4
    bp = (Rd - R)*(grd - gr)/mu
    
    if bp == 0: P = 0
    else: P = np.min([0, -ap])/bp
    u_des = np.array([At, P, Q])
    # return u_des

    # closed form expression after being optimised
    # x = r v v_dot here, not the full state
    # c3bf original with backstepping from R to P
    def c3bf_backstep():
        # depacking relevant information
        r_ob = np.array(x_ob[:3])
        v_ob = np.array(x_ob[3:6])
        a_ob = np.array(x_ob[6:])

        # relative terms
        # p_rel_dot = v_rel; v_rel_dot = a_rel 
        p_rel = r_ob - r
        v_rel = v_ob - v
        a_rel = a_ob - v_dot

        # cbf h(x,t)
        h1 = np.dot(p_rel, v_rel)
        h2 = np.sqrt(np.linalg.norm(p_rel)**2 - rad**2)
        h3 = -((Rd - R)**2)/(2*mu_e)
        h4 = np.linalg.norm(v_rel)
        h = h1 + h2*h4 + h3

        # cbf_dot
        hd1 = h4**2
        hd2 = np.dot(p_rel, a_rel)
        hd3 = np.dot(v_rel, a_rel)*h2/h4
        hd4 = h1*h4/h2
        hd5 = -(Rd - R)*(frd - fr + (grd - gr)*u_des[1])/mu_e # u_des[1] = P
        h_dot = hd1 + hd2 + hd3 + hd4 + hd5
        
        # Lgh
        Lg_h = np.array([-2*(mu*((r_ob[0] - n)*(v_ob[0] - vt*cos(psi)*cos(theta)) + (r_ob[1] - e)*(v_ob[1] - vt*sin(psi)*cos(theta)) + (r_ob[2] - d)*(v_ob[2] + vt*sin(theta)) + np.sqrt((v_ob[0] - vt*cos(psi)*cos(theta))**2 + (v_ob[1] - vt*sin(psi)*cos(theta))**2 + (v_ob[2] + vt*sin(theta))**2)*(-rad**2 + (r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2)**0.5)*vt**2 - (2*g*sin(phi)*cos(theta) + (sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) + Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) - 2*ag[1]) - (-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) + Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) - 2*ag[0]) - (2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*sin(phi)*cos(theta))**2/8)/(mu*vt**3) + (2*mu*((r_ob[0] - n)*(v_ob[0] - vt*cos(psi)*cos(theta)) + (r_ob[1] - e)*(v_ob[1] - vt*sin(psi)*cos(theta)) + (r_ob[2] - d)*(v_ob[2] + vt*sin(theta)) + np.sqrt((v_ob[0] - vt*cos(psi)*cos(theta))**2 + (v_ob[1] - vt*sin(psi)*cos(theta))**2 + (v_ob[2] + vt*sin(theta))**2)*(-rad**2 + (r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2)**0.5)*vt + mu*(-(r_ob[0] - n)*cos(psi)*cos(theta) - (r_ob[1] - e)*sin(psi)*cos(theta) + (r_ob[2] - d)*sin(theta) + (-(v_ob[0] - vt*cos(psi)*cos(theta))*cos(psi)*cos(theta) - (v_ob[1] - vt*sin(psi)*cos(theta))*sin(psi)*cos(theta) + (v_ob[2] + vt*sin(theta))*sin(theta))*(-rad**2 + (r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2)**0.5/np.sqrt((v_ob[0] - vt*cos(psi)*cos(theta))**2 + (v_ob[1] - vt*sin(psi)*cos(theta))**2 + (v_ob[2] + vt*sin(theta))**2))*vt**2 - (-2*(2*Kr[2,2]*sin(theta) + Kv[2,2]*sin(theta))*sin(phi)*cos(theta) - 2*(2*Kr[0,0]*cos(phi)*cos(psi) + Kv[0,0]*cos(psi)*cos(theta))*(-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi)) + 2*(2*Kr[1,1]*sin(psi)*cos(phi) + Kv[1,1]*sin(psi)*cos(theta))*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi)))*(2*g*sin(phi)*cos(theta) + (sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) + Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) - 2*ag[1]) - (-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) + Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) - 2*ag[0]) - (2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*sin(phi)*cos(theta))/8)/(mu*vt**2), -(2*g*sin(phi)*cos(theta) + (sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) + Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) - 2*ag[1]) - (-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) + Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) - 2*ag[0]) - (2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*sin(phi)*cos(theta))*(4*g*cos(phi)*cos(theta) + 4*Kr[0,0]*(-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*vt*sin(phi)*cos(psi) - 4*Kr[1,1]*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*vt*sin(phi)*sin(psi) - 2*(-sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*(2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) + Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) - 2*ag[0]) + 2*(-sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*(2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) + Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) - 2*ag[1]) - 2*(2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*cos(phi)*cos(theta))/(8*mu*vt**2), (mu*(((v_ob[0] - vt*cos(psi)*cos(theta))*vt*sin(psi)*cos(theta) - (v_ob[1] - vt*sin(psi)*cos(theta))*vt*cos(psi)*cos(theta))*(-rad**2 + (r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2)**0.5/np.sqrt((v_ob[0] - vt*cos(psi)*cos(theta))**2 + (v_ob[1] - vt*sin(psi)*cos(theta))**2 + (v_ob[2] + vt*sin(theta))**2) + (r_ob[0] - n)*vt*sin(psi)*cos(theta) - (r_ob[1] - e)*vt*cos(psi)*cos(theta))*vt**2 - (2*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(2*Kr[1,1]*vt*cos(phi)*cos(psi) + Kv[1,1]*vt*cos(psi)*cos(theta)) - 2*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) + Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) - 2*ag[0]) - 2*(-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(-2*Kr[0,0]*vt*sin(psi)*cos(phi) - Kv[0,0]*vt*sin(psi)*cos(theta)) + 2*(sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi))*(2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) + Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) - 2*ag[1]))*(2*g*sin(phi)*cos(theta) + (sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) + Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) - 2*ag[1]) - (-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) + Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) - 2*ag[0]) - (2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*sin(phi)*cos(theta))/8)*sin(phi)/(mu*vt**2*cos(theta)) + (mu*((r_ob[0] - n)*vt*sin(theta)*cos(psi) + (r_ob[1] - e)*vt*sin(psi)*sin(theta) + (r_ob[2] - d)*vt*cos(theta) + ((v_ob[0] - vt*cos(psi)*cos(theta))*vt*sin(theta)*cos(psi) + (v_ob[1] - vt*sin(psi)*cos(theta))*vt*sin(psi)*sin(theta) + (v_ob[2] + vt*sin(theta))*vt*cos(theta))*(-rad**2 + (r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2)**0.5/np.sqrt((v_ob[0] - vt*cos(psi)*cos(theta))**2 + (v_ob[1] - vt*sin(psi)*cos(theta))**2 + (v_ob[2] + vt*sin(theta))**2))*vt**2 - (2*g*sin(phi)*cos(theta) + (sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) + Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) - 2*ag[1]) - (-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) + Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) - 2*ag[0]) - (2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*sin(phi)*cos(theta))*(-4*g*sin(phi)*sin(theta) + 2*Kv[0,0]*(-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*vt*sin(theta)*cos(psi) - 2*Kv[1,1]*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*vt*sin(psi)*sin(theta) - 2*(2*Kr[2,2]*vt*cos(theta) + Kv[2,2]*vt*cos(theta))*sin(phi)*cos(theta) + 2*(2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) + Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) - 2*ag[0])*sin(phi)*cos(psi)*cos(theta) + 2*(2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) + Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) - 2*ag[1])*sin(phi)*sin(psi)*cos(theta) + 2*(2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*sin(phi)*sin(theta))/8)*cos(phi)/(mu*vt**2) - (2*g*sin(phi)*cos(theta) + (sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) + Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) - 2*ag[1]) - (-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) + Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) - 2*ag[0]) - (2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*sin(phi)*cos(theta))*(4*g*cos(phi)*cos(theta) + 4*Kr[0,0]*(-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*vt*sin(phi)*cos(psi) - 4*Kr[1,1]*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*vt*sin(phi)*sin(psi) - 2*(-sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*(2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) + Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) - 2*ag[0]) + 2*(-sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*(2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) + Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) - 2*ag[1]) - 2*(2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*cos(phi)*cos(theta))*sin(phi)*tan(theta)/(8*mu*vt**2)])    

        # optimisation solution in closed form
        a = h_dot + h # class K function alpha(r) = r
        b = np.dot(W, Lg_h)
        coeff = 1
        mod_b = np.linalg.norm(b)
        if mod_b == 0: lamb = 0
        # else: lamb = np.max((0, -a/mod_b))/mod_b
        else: lamb = np.log(1 + np.exp(-coeff*a/mod_b))/mod_b/coeff
        u_safe = u_des + lamb*np.dot(W, b)
        
        return u_safe
    
    # cbf as described in paper with backstepping from R to P
    def cbf_backstep():
        # depacking relevant information
        r_ob = np.array(x_ob[:3])
        v_ob = np.array(x_ob[3:6])
        a_ob = np.array(x_ob[6:])

        # relative terms
        # p_rel_dot = v_rel; v_rel_dot = a_rel 
        p_rel = r_ob - r
        v_rel = v_ob - v
        a_rel = a_ob - v_dot

        # cbf h(x,t)
        h1 = np.linalg.norm(p_rel)
        h2 = np.dot(p_rel, v_rel)/h1
        h3 = -((Rd - R)**2)/(2*mu_e) - rad
        h = h1 + 10*h2 + h3

        # cbf_dot
        hd1 = (np.dot(p_rel, a_rel) + np.linalg.norm(v_rel)**2)/h1
        hd2 = -(h2**2)/h1
        hd3 = -(Rd - R)*(frd - fr + (grd - gr)*u_des[1])/mu_e # u_des[1] = P
        h_dot = h2 + hd1 + hd2 + hd3
        
        # Lgh
        Lg_h = np.array([-2*(mu_e*(-rad + np.sqrt((r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2))*np.sqrt((r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2)*vt**2 + 10*mu_e*((r_ob[0] - n)*(v_ob[0] - vt*cos(psi)*cos(theta)) + (r_ob[1] - e)*(v_ob[1] - vt*sin(psi)*cos(theta)) + (r_ob[2] - d)*(v_ob[2] + vt*sin(theta)))*vt**2 - np.sqrt((r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2)*(2*g*sin(phi)*cos(theta) - (sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(-2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) - Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) + 2*ag[1]) + (-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(-2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) - Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) + 2*ag[0]) - (2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*sin(phi)*cos(theta))**2/8)/(mu_e*np.sqrt((r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2)*vt**3) + (2*mu_e*(-rad + np.sqrt((r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2))*np.sqrt((r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2)*vt + 20*mu_e*((r_ob[0] - n)*(v_ob[0] - vt*cos(psi)*cos(theta)) + (r_ob[1] - e)*(v_ob[1] - vt*sin(psi)*cos(theta)) + (r_ob[2] - d)*(v_ob[2] + vt*sin(theta)))*vt + 10*mu_e*(-(r_ob[0] - n)*cos(psi)*cos(theta) - (r_ob[1] - e)*sin(psi)*cos(theta) + (r_ob[2] - d)*sin(theta))*vt**2 - (-2*(2*Kr[2,2]*sin(theta) + Kv[2,2]*sin(theta))*sin(phi)*cos(theta) + 2*(-2*Kr[0,0]*cos(phi)*cos(psi) - Kv[0,0]*cos(psi)*cos(theta))*(-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi)) - 2*(-2*Kr[1,1]*sin(psi)*cos(phi) - Kv[1,1]*sin(psi)*cos(theta))*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi)))*np.sqrt((r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2)*(2*g*sin(phi)*cos(theta) - (sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(-2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) - Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) + 2*ag[1]) + (-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(-2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) - Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) + 2*ag[0]) - (2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*sin(phi)*cos(theta))/8)/(mu_e*np.sqrt((r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2)*vt**2), -(2*g*sin(phi)*cos(theta) - (sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(-2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) - Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) + 2*ag[1]) + (-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(-2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) - Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) + 2*ag[0]) - (2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*sin(phi)*cos(theta))*(4*g*cos(phi)*cos(theta) + 4*Kr[0,0]*(-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*vt*sin(phi)*cos(psi) - 4*Kr[1,1]*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*vt*sin(phi)*sin(psi) + 2*(-sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*(-2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) - Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) + 2*ag[0]) - 2*(-sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*(-2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) - Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) + 2*ag[1]) - 2*(2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*cos(phi)*cos(theta))/(8*mu_e*vt**2), (10*mu_e*((r_ob[0] - n)*vt*sin(psi)*cos(theta) - (r_ob[1] - e)*vt*cos(psi)*cos(theta))*vt**2 - np.sqrt((r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2)*(-2*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(-2*Kr[1,1]*vt*cos(phi)*cos(psi) - Kv[1,1]*vt*cos(psi)*cos(theta)) + 2*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(-2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) - Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) + 2*ag[0]) + 2*(-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(2*Kr[0,0]*vt*sin(psi)*cos(phi) + Kv[0,0]*vt*sin(psi)*cos(theta)) - 2*(sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi))*(-2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) - Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) + 2*ag[1]))*(2*g*sin(phi)*cos(theta) - (sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(-2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) - Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) + 2*ag[1]) + (-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(-2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) - Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) + 2*ag[0]) - (2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*sin(phi)*cos(theta))/8)*sin(phi)/(mu_e*np.sqrt((r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2)*vt**2*cos(theta)) + (10*mu_e*((r_ob[0] - n)*vt*sin(theta)*cos(psi) + (r_ob[1] - e)*vt*sin(psi)*sin(theta) + (r_ob[2] - d)*vt*cos(theta))*vt**2 - np.sqrt((r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2)*(2*g*sin(phi)*cos(theta) - (sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(-2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) - Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) + 2*ag[1]) + (-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(-2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) - Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) + 2*ag[0]) - (2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*sin(phi)*cos(theta))*(-4*g*sin(phi)*sin(theta) + 2*Kv[0,0]*(-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*vt*sin(theta)*cos(psi) - 2*Kv[1,1]*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*vt*sin(psi)*sin(theta) - 2*(2*Kr[2,2]*vt*cos(theta) + Kv[2,2]*vt*cos(theta))*sin(phi)*cos(theta) - 2*(-2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) - Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) + 2*ag[0])*sin(phi)*cos(psi)*cos(theta) - 2*(-2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) - Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) + 2*ag[1])*sin(phi)*sin(psi)*cos(theta) + 2*(2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*sin(phi)*sin(theta))/8)*cos(phi)/(mu_e*np.sqrt((r_ob[0] - n)**2 + (r_ob[1] - e)**2 + (r_ob[2] - d)**2)*vt**2) - (2*g*sin(phi)*cos(theta) - (sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*(-2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) - Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) + 2*ag[1]) + (-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*(-2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) - Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) + 2*ag[0]) - (2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*sin(phi)*cos(theta))*(4*g*cos(phi)*cos(theta) + 4*Kr[0,0]*(-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*vt*sin(phi)*cos(psi) - 4*Kr[1,1]*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*vt*sin(phi)*sin(psi) + 2*(-sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*(-2*Kr[0,0]*(vt*cos(phi)*cos(psi) - vg[0]) - Kv[0,0]*(Kr[0,0]*(n - rg[0]) + vt*cos(psi)*cos(theta) - vg[0]) + 2*ag[0]) - 2*(-sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*(-2*Kr[1,1]*(vt*sin(psi)*cos(phi) - vg[1]) - Kv[1,1]*(Kr[1,1]*(e - rg[1]) + vt*sin(psi)*cos(theta) - vg[1]) + 2*ag[1]) - 2*(2*Kr[2,2]*(vt*sin(theta) + vg[2]) + Kv[2,2]*(-Kr[2,2]*(d - rg[2]) + vt*sin(theta) + vg[2]) + 2*ag[2])*cos(phi)*cos(theta))*sin(phi)*tan(theta)/(8*mu_e*vt**2)])

        # optimisation solution in closed form
        a = h_dot + h # class K function alpha(r) = r
        b = np.dot(W, Lg_h)
        coeff = 1
        mod_b = np.linalg.norm(b)
        if mod_b == 0: lamb = 0
        else: lamb = np.max((0, -a/mod_b))/mod_b
        # else: lamb = np.log(1 + np.exp(-coeff*a/mod_b))/mod_b/coeff
        u_safe = u_des + lamb*np.dot(W, b)
        
        return u_safe
    
    u_safe = c3bf_backstep()
    return u_safe
    
# final simulation with plotted graphs
def sim(x0, traj_self, traj_ob, t_params, con_params):
    # time depacking
    t_i, t_f, delta = t_params
    t_eval = np.arange(t_i,t_f, delta)
                       
    # solver output
    sol = solve_ivp(x_dot, [t_i,t_f], x0, t_eval=t_eval, args=(traj_self, traj_ob, delta, con_params))

    # displaying coordinates of the aircracft
    plt.figure()
    ax = plt.axes(projection = '3d')

    ax.plot(traj_self[:, 0], traj_self[:, 1], traj_self[:, 2], label='Desired', linestyle='--')
    ax.plot(traj_ob[:, 0], traj_ob[:, 1], traj_ob[:, 2], label='Obstacle', linestyle='--')
    ax.plot(sol.y[0, :], sol.y[1, :], sol.y[2, :], label='Actual')

    ax.set_xlim([-3000, 3000])
    ax.set_zlim([-500, 500])

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

# -----------MAIN CODE-----------
# initial state (n e d phi theta psi vt), where/how the plane spawns
x0 = np.array([0,0,0,0,0,0.5*np.pi,170]) 

# time params
delta = 0.001
t_i = 0
t_f = 50
t_params = np.array([t_i, t_f, delta])

# control params
Kr = 0.1*np.diag([1,1,1]) # both should be positive definite
Kv = 0.6*np.diag([1,1,1])
W = np.diag([6, 0.6, 0.1])
mu = 10e-5 # scale factor for backstepper
mu_e = 10e-4 # scale factor for safety
rad_ob = 100 # obstacle radius
con_params = [Kr, Kv, mu, rad_ob, mu_e, W]

# linear trajectory generators given initial position and velocity
r0_self = np.array([0,0,0])
v0_self = np.array([0,170,0])

r0_ob_oncoming = np.array([0,5000,-10])
v0_ob_oncoming = np.array([0,-170,0])

r0_ob_side = np.array([-3000,0,0])
v0_ob_side = np.array([130,170,0])

way_path = way_gen(r0_self, v0_self)
way_ob_oncoming = way_gen(r0_ob_oncoming, v0_ob_oncoming)
way_ob_side = way_gen(r0_ob_side, v0_ob_side)

traj_path = trajectory(way_path, delta, t_i, t_f)
traj_ob_oncoming = trajectory(way_ob_oncoming, delta, t_i, t_f)
traj_ob_side = trajectory(way_ob_side, delta, t_i, t_f)

# simulate
sim(x0, traj_path, traj_ob_oncoming, t_params, con_params)

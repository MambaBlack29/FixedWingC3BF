# Fixed Wing UAV with C3Bf

## 1. Fixed Wing Kinematics
- The kinematic model of the fixed wing follows the 3D Dubins model for a VISTA aircraft
- The state space consists of 7 independent variables, namely:
    - `n` - North
    - `e` - East
    - `d` - Down (technically positive is up)
    - `phi` - Roll angle
    - `theta` - Pitch angle
    - `psi` - Yaw angle
    - `vt` - Linear forward velocity along body `x` axis
- There are 3 inputs to the system
    - `At` - Linear acceleration along body `x` axis
    - `P` - Roll rate
    - `Q` - Pitch rate

## 2. Base Controller
- The base controller is a velocity tracking controller which follows a specified trajectory
- The trajectory generated in the code is a cubic spline made using `scipy.interpolate`
- The kinematics are arranged in a manner such that a basic error minimising approach leads to controll inputs `At`, `Q`, and `R`, which are the linear acceleration, pitch rate and yaw rate respectively
- Since there is no direct control over `R`, a CLF backstepping controller is created which backsteps from `R` (desired) to `P`

## 3. Control Barrier Function (Collision Cone)
- The CBF candidate used here is a Collision Cone Controll Barrier Function
- A collision cone is created from the boundary of the obstacle to the UAV
- The intuition behind the C3BF is just that the relative velocity should lie **outside** the collision cone
- In the above code, the closed form expression of the final safe controller is used using KKT conditions on the Quadratic Problem associated with CBFs

## 4. How to run?
- Directly run the `fixed_wing_controller.py` file using `python` or `python3`
- Ensure you have `numpy`, `scipy` and `matplotlib` installed
- I have included the codes which were used to generate **symbolically** certain expressions used in the final code
- To use them, ensure you have `sympy` installed
- **NOTE** - All the Symbolic codes may take upto 30-40 minutes to run, since `sympy.simplify` has been used for *very* large expressions
- https://arxiv.org/pdf/2403.02508
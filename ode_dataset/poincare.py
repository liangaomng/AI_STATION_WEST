
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
omega = 1 # 选择固有频率为1
def spring_pendulum(t, Z):
    x, y = Z
    dxdt = y
    dydt = -omega**2 * x
    return [dxdt, dydt]
def cross_x_zero(t, Z):
    return Z[0]
cross_x_zero.terminal = True
cross_x_zero.direction = -1 # 只考虑从正到负的交叉
# 初始条件
z0 = [0.01, 1]
poincare_points = []
for _ in range(100):
    sol = solve_ivp(spring_pendulum, (0, 100), z0, events=cross_x_zero, dense_output=True)
if hasattr(sol, 't_events') and len(sol.t_events[0]) > 0:
    y_cross = sol.y_events[0][0][1]
    poincare_points.append(y_cross)
    z0 = sol.y_events[0][0]
    z0[0] = 0.01

plt.scatter(np.zeros(len(poincare_points)), poincare_points, s=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Poincaré Map for x=0')
plt.show()
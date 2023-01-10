# %%
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# %%


def ode_richardson(t, z, k, l, alpha, beta, g, h):
    dx = k * z[1] - alpha * z[0] + g
    dy = l * z[0] - beta * z[1] + h

    return [dx, dy]


# %%
k, l = 1, 1
alpha, beta = 2, 2
g, h = 2, 3

# %%
x0, y0 = 10.0, -4.0
Tend = 7.0

sol = solve_ivp(
    ode_richardson, 
    t_span=[0, Tend],
    y0=[x0, y0],
    args=(k, l, alpha, beta, g, h),
    dense_output=True
)

# %%
plt.plot(sol["t"], sol["y"][0])
plt.plot(sol["t"], sol["y"][1])
plt.show()

# %%
plt.plot(sol["y"][0], sol["y"][1])
plt.show()

# %% ここには微係数のベクトル場を描く。

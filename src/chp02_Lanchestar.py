# %%
from scipy.integrate import solve_ivp
import numpy as np
# import seaborn
import matplotlib.pyplot as plt
# import japanize-matplotlib

# %%


def ode_Lan1(t, z, cx, cy):
    dx = -cy  # Y軍の武器の性能
    dy = -cx  # X軍の武器の性能

    return [dx, dy]


def ode_Lan2(t, z, cx, cy):
    dx = -cy * z[1]
    dy = -cx * z[0]

    return [dx, dy]
# %%


def Lanchestar_func(x0=5.0, y0=7.0, cx=1.0, cy=1.0, tend=5.0, law=1):
    if law == 1:
        sol = solve_ivp(
            ode_Lan1,
            t_span=[0, tend],  # 積分範囲。
            y0=[x0, y0],  # 初期値。
            args=(cx, cy,),  # solve_ivpにわたすパラメータ。
            dense_output=True  # なめらかな数値解を渡すための奴(深く考えるな)
            )

        tcl = np.linspace(0, tend, 100)
        ycl = sol.sol(tcl)

        return tcl, ycl
    elif law == 2:
        sol = solve_ivp(
            ode_Lan2,
            t_span=[0, tend],  # 積分範囲。
            y0=[x0, y0],  # 初期値。
            args=(cx, cy,),  # solve_ivpにわたすパラメータ。
            dense_output=True  # なめらかな数値解を渡すための奴(深く考えるな)
            )

        tcl = np.linspace(0, tend, 100)
        ycl = sol.sol(tcl)

        return tcl, ycl


# %% calc
# 1st law
tcl_1, ycl_1 = Lanchestar_func(cx=1.0, cy=1.0, law=1, tend=5.0)
tcl_2, ycl_2 = Lanchestar_func(cx=3.0, cy=1.0, law=1, tend=2.5)

# 2nd law
tcl_3, ycl_3 = Lanchestar_func(tend=0.8, cx=1.0, cy=1.0, law=2)
tcl_4, ycl_4 = Lanchestar_func(tend=0.6, cx=3.0, cy=1.0, law=2)


# %%
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

ax[0, 0].set_title("Lanchestar 1st law: cx=1.0, cy=1.0")
ax[0, 0].plot(tcl_1, ycl_1[0].T)
ax[0, 0].plot(tcl_1, ycl_1[1].T)

ax[0, 1].set_title("Lanchestar 1st law: cx=3.0, cy=1.0")
ax[0, 1].plot(tcl_2, ycl_2[0].T)
ax[0, 1].plot(tcl_2, ycl_2[1].T)

ax[1, 0].set_title("Lanchestar 2nd law: cx=1.0, cy=1.0")
ax[1, 0].plot(tcl_3, ycl_3[0])
ax[1, 0].plot(tcl_3, ycl_3[1])

ax[1, 1].set_title("Lanchestar 2nd law: cx=3.0, cy=1.0")
ax[1, 1].plot(tcl_4, ycl_4[0])
ax[1, 1].plot(tcl_4, ycl_4[1])

plt.show()
# %%

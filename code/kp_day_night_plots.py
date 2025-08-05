import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from functions import maxIndex
from astropy import constants as const

mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern Serif",
    "font.size": 35,
    "figure.figsize": [20, 15],
    "axes.facecolor": (1.0 ,1.0, 1.0, 1.0),
    "savefig.facecolor": (1.0 ,1.0, 1.0, 1.0)
})

home_path = os.environ["HOME"]

local_path = home_path + "/exoplanet_atmospheres/code"
local_images = home_path + "/exoplanet_atmospheres/images"

Kp = 198.04

day_data = np.load(os.path.join(local_path, "day-night_kp_day.npz"))
night_data = np.load(os.path.join(local_path, "day-night_kp_night.npz"))

n_exposure = 300

op_pre_day = np.linspace(0.33, 0.43, n_exposure)  # time/period
op_post_day = np.linspace(0.55, 0.66, n_exposure)  # time/period

op_pre_night = np.linspace(-0.12, -0.04, n_exposure)  # time/period
op_post_night = np.linspace(0.04, 0.12, n_exposure)  # time/period

kp_day_day = day_data['day']
kp_night_day = day_data['night']
kp_tot_day = day_data['tot']

cc_day_pre_day = day_data['cc_day_pre']
cc_night_pre_day= day_data['cc_night_pre']
cc_tot_pre_day = day_data['cc_tot_pre']

cc_day_post_day= day_data['cc_day_post']
cc_night_post_day= day_data['cc_night_post']
cc_tot_post_day = day_data['cc_tot_post']

kp_day_night = night_data['day']
kp_night_night = night_data['night']
kp_tot_night = night_data['tot']

cc_day_pre_night = night_data['cc_day_pre']
cc_night_pre_night = night_data['cc_night_pre']
cc_tot_pre_night = night_data['cc_tot_pre']

cc_day_post_night = night_data['cc_day_post']
cc_night_post_night= night_data['cc_night_post']
cc_tot_post_night = night_data['cc_tot_post']

vsys = np.linspace(-200,200, 1001)
vsys_kp = np.linspace(-30, 30, 1001)
K = np.linspace(Kp - 85, Kp + 85, 1001)

fig, ax = plt.subplots()
ax.pcolormesh(vsys_kp, K, kp_tot_day)
ax.set_title(r'$K_p$ Total')
fig.supylabel(r"$K_p$ (km/s)")
fig.supxlabel(r"v$_{\hbox{sys}}$ (km/s)")

ax.axhline(
    Kp,
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Actual $K_p$ = " + f"{Kp:.2f}km/s",
)
ax.axvline(
    0,
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Actual v$_{\hbox{sys}}$ = " + f"{0}km/s",
)
ax.axhline(
    K[maxIndex(kp_tot_day)[0]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured $K_p$ = " + f"{K[maxIndex(kp_tot_day)[0]]:.2f}km/s",
)
ax.axvline(
    vsys_kp[maxIndex(kp_tot_day)[1]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured v$_{\hbox{sys}}$ = "
    + f"{vsys_kp[maxIndex(kp_tot_day)[1]]:.2f}km/s",
)

ax.legend(loc='upper left')
plt.savefig(os.path.join(local_images, "kp_tot_day.png"))



fig, ax = plt.subplots(2)
ax[0].pcolormesh(vsys_kp, K, kp_day_day)
fig.supylabel(r"$K_p$ (km/s)")
fig.supxlabel(r"v$_{\hbox{sys}}$ (km/s)")
ax[0].set_title(r'$K_p$ Day')
ax[0].axhline(
    Kp,
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Actual $K_p$ = " + f"{Kp:.2f}km/s",
)
ax[0].axvline(
    0,
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Actual v$_{\hbox{sys}}$ = " + f"{0}km/s",
)
ax[0].axhline(
    K[maxIndex(kp_day_day)[0]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured $K_p$ = " + f"{K[maxIndex(kp_day_day)[0]]:.2f}km/s",
)
ax[0].axvline(
    vsys_kp[maxIndex(kp_day_day)[1]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured v$_{\hbox{sys}}$ = "
    + f"{vsys_kp[maxIndex(kp_day_day)[1]]:.2f}km/s",
)

ax[1].pcolormesh(vsys_kp, K, kp_night_day)
fig.supylabel(r"$K_p$ (km/s)")
fig.supxlabel(r"v$_{\hbox{sys}}$ (km/s)")
ax[1].set_title(r'$K_p$ Night')

ax[1].axhline(
    Kp,
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Actual $K_p$ = " + f"{Kp:.2f}km/s",
)
ax[1].axvline(
    0,
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Actual v$_{\hbox{sys}}$ = " + f"{0}km/s",
)
ax[1].axhline(
    K[maxIndex(kp_night_day)[0]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured $K_p$ = " + f"{K[maxIndex(kp_night_day)[0]]:.2f}km/s",
)
ax[1].axvline(
    vsys_kp[maxIndex(kp_night_day)[1]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured v$_{\hbox{sys}}$ = "
    + f"{vsys_kp[maxIndex(kp_night_day)[1]]:.2f}km/s",
)
ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')
plt.savefig(os.path.join(local_images, "kp_day-night_day.png"))


fig, ax = plt.subplots()
fig.supxlabel(r"v$_{\hbox{sys}}$ (km/s)")
fig.supylabel(r"Orbital Phase ($\phi$)")
ax.pcolormesh(vsys, op_pre_day, cc_tot_pre_day)
ax.set_title(r'CC Total')
ax.plot(K[maxIndex(kp_tot_day)][0] * np.sin(2 * np.pi * op_pre_day), op_pre_day, color='red', ls='--', label = '$K_p$ = ' + f'{ K[maxIndex(kp_tot_day)][0]:.2f}')
ax.legend(loc= 'upper left')
plt.savefig(os.path.join(local_images, "cc_tot_pre_day.png"))

fig, ax = plt.subplots(2)
fig.supxlabel(r"v$_{\hbox{sys}}$ (km/s)")
fig.supylabel(r"Orbital Phase ($\phi$)")
ax[0].pcolormesh(vsys, op_pre_day, cc_day_pre_day)
ax[0].set_title(r'CC Day')
ax[0].plot(K[maxIndex(kp_day_day)][0] * np.sin(2 * np.pi * op_pre_day), op_pre_day, color='red', ls='--', label = '$K_p$ = ' + f'{ K[maxIndex(kp_day_day)][0]:.2f}')
ax[1].pcolormesh(vsys, op_pre_day, cc_night_pre_day)
ax[1].set_title(r'CC Night')
ax[1].plot(K[maxIndex(kp_night_day)][0] * np.sin(2 * np.pi * op_pre_day), op_pre_day, color='red', ls='--', label = '$K_p$ = ' + f'{ K[maxIndex(kp_night_day)][0]:.2f}')
ax[0].legend(loc ='upper left')
ax[1].legend(loc ='upper left')
plt.savefig(os.path.join(local_images, "cc_day-night_pre_day.png"))


fig, ax = plt.subplots()
fig.supxlabel(r"v$_{\hbox{sys}}$ (km/s)")
fig.supylabel(r"Orbital Phase ($\phi$)")
ax.pcolormesh(vsys, op_post_day, cc_tot_post_day)
ax.set_title(r'CC Total')
ax.plot(K[maxIndex(kp_tot_day)][0] * np.sin(2 * np.pi * op_post_day), op_post_day, color='red', ls='--', label = '$K_p$ = ' + f'{ K[maxIndex(kp_tot_day)][0]:.2f}')
ax.legend(loc= 'upper left')
plt.savefig(os.path.join(local_images, "cc_tot_post_day.png"))

fig, ax = plt.subplots(2)
fig.supxlabel(r"v$_{\hbox{sys}}$ (km/s)")
fig.supylabel(r"Orbital Phase ($\phi$)")
ax[0].pcolormesh(vsys, op_post_day, cc_day_post_day)
ax[0].set_title(r'CC Day')
ax[0].plot(K[maxIndex(kp_day_day)][0] * np.sin(2 * np.pi * op_post_day), op_post_day, color='red', ls='--', label = '$K_p$ = ' + f'{ K[maxIndex(kp_day_day)][0]:.2f}')
ax[1].pcolormesh(vsys, op_post_day, cc_night_post_day)
ax[1].set_title(r'CC Night')
ax[1].plot(K[maxIndex(kp_night_day)][0] * np.sin(2 * np.pi * op_post_day), op_post_day, color='red', ls='--', label = '$K_p$ = ' + f'{ K[maxIndex(kp_night_day)][0]:.2f}')
ax[0].legend(loc ='upper left')
ax[1].legend(loc ='upper left')
plt.savefig(os.path.join(local_images, "cc_day-night_post_day.png"))

fig, ax = plt.subplots()
ax.pcolormesh(vsys_kp, K, kp_tot_night)
ax.set_title(r'$K_p$ Total')
fig.supylabel(r"$K_p$ (km/s)")
fig.supxlabel(r"v$_{\hbox{sys}}$ (km/s)")

ax.axhline(
    Kp,
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Actual $K_p$ = " + f"{Kp:.2f}km/s",
)
ax.axvline(
    0,
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Actual v$_{\hbox{sys}}$ = " + f"{0}km/s",
)
ax.axhline(
    K[maxIndex(kp_tot_night)[0]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured $K_p$ = " + f"{K[maxIndex(kp_tot_night)[0]]:.2f}km/s",
)
ax.axvline(
    vsys_kp[maxIndex(kp_tot_night)[1]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured v$_{\hbox{sys}}$ = "
    + f"{vsys_kp[maxIndex(kp_tot_night)[1]]:.2f}km/s",
)

ax.legend(loc='upper left')
plt.savefig(os.path.join(local_images, "kp_tot_night.png"))



fig, ax = plt.subplots(2)
ax[0].pcolormesh(vsys_kp, K, kp_day_night)
fig.supylabel(r"$K_p$ (km/s)")
fig.supxlabel(r"v$_{\hbox{sys}}$ (km/s)")
ax[0].set_title(r'$K_p$ Day')
ax[1].set_title(r'$K_p$ Night')

ax[0].axhline(
    Kp,
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Actual $K_p$ = " + f"{Kp:.2f}km/s",
)
ax[0].axvline(
    0,
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Actual v$_{\hbox{sys}}$ = " + f"{0}km/s",
)
ax[0].axhline(
    K[maxIndex(kp_day_night)[0]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured $K_p$ = " + f"{K[maxIndex(kp_day_night)[0]]:.2f}km/s",
)
ax[0].axvline(
    vsys_kp[maxIndex(kp_day_night)[1]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured v$_{\hbox{sys}}$ = "
    + f"{vsys_kp[maxIndex(kp_day_night)[1]]:.2f}km/s",
)

ax[1].pcolormesh(vsys_kp, K, kp_night_night)
fig.supylabel(r"$K_p$ (km/s)")
fig.supxlabel(r"v$_{\hbox{sys}}$ (km/s)")
ax[1].set_title(r'$K_p$ Night')
ax[1].axhline(
    Kp,
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Actual $K_p$ = " + f"{Kp:.2f}km/s",
)
ax[1].axvline(
    0,
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Actual v$_{\hbox{sys}}$ = " + f"{0}km/s",
)
ax[1].axhline(
    K[maxIndex(kp_night_night)[0]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured $K_p$ = " + f"{K[maxIndex(kp_night_night)[0]]:.2f}km/s",
)
ax[1].axvline(
    vsys_kp[maxIndex(kp_night_night)[1]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured v$_{\hbox{sys}}$ = "
    + f"{vsys_kp[maxIndex(kp_night_night)[1]]:.2f}km/s",
)
ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')
plt.savefig(os.path.join(local_images, "kp_day-night_night.png"))


fig, ax = plt.subplots()
fig.supxlabel(r"v$_{\hbox{sys}}$ (km/s)")
fig.supylabel(r"Orbital Phase ($\phi$)")
ax.set_title(r'CC Total')
ax.pcolormesh(vsys, op_pre_night, cc_tot_pre_night)
ax.plot(K[maxIndex(kp_tot_night)][0] * np.sin(2 * np.pi * op_pre_night), op_pre_night, color='red', ls='--', label = '$K_p$ = ' + f'{ K[maxIndex(kp_tot_night)][0]:.2f}')
ax.legend(loc= 'upper left')
plt.savefig(os.path.join(local_images, "cc_tot_pre_night.png"))

fig, ax = plt.subplots(2)
fig.supxlabel(r"v$_{\hbox{sys}}$ (km/s)")
fig.supylabel(r"Orbital Phase ($\phi$)")
ax[0].set_title(r'CC Day')
ax[1].set_title(r'CC Night')
ax[0].pcolormesh(vsys, op_pre_night, cc_day_pre_night)
ax[0].plot(K[maxIndex(kp_day_night)][0] * np.sin(2 * np.pi * op_pre_night), op_pre_night, color='red', ls='--', label = '$K_p$ = ' + f'{ K[maxIndex(kp_day_night)][0]:.2f}')
ax[1].pcolormesh(vsys, op_pre_night, cc_night_pre_night)
ax[1].plot(K[maxIndex(kp_night_night)][0] * np.sin(2 * np.pi * op_pre_night), op_pre_night, color='red', ls='--', label = '$K_p$ = ' + f'{ K[maxIndex(kp_night_night)][0]:.2f}')
ax[0].legend(loc ='upper left')
ax[1].legend(loc ='upper left')
plt.savefig(os.path.join(local_images, "cc_day-night_pre_night.png"))


fig, ax = plt.subplots()
fig.supxlabel(r"v$_{\hbox{sys}}$ (km/s)")
fig.supylabel(r"Orbital Phase ($\phi$)")
ax.set_title(r'CC Total')
ax.pcolormesh(vsys, op_post_night, cc_tot_post_night)
ax.plot(K[maxIndex(kp_tot_night)][0] * np.sin(2 * np.pi * op_post_night), op_post_night, color='red', ls='--', label = '$K_p$ = ' + f'{ K[maxIndex(kp_tot_night)][0]:.2f}')
ax.legend(loc= 'upper left')
plt.savefig(os.path.join(local_images, "cc_tot_post_night.png"))

fig, ax = plt.subplots(2)
fig.supxlabel(r"v$_{\hbox{sys}}$ (km/s)")
fig.supylabel(r"Orbital Phase ($\phi$)")
ax[0].set_title(r'CC Day')
ax[1].set_title(r'CC Night')
ax[0].pcolormesh(vsys, op_post_night, cc_day_post_night)
ax[0].plot(K[maxIndex(kp_day_night)][0] * np.sin(2 * np.pi * op_post_night), op_post_night, color='red', ls='--', label = '$K_p$ = ' + f'{ K[maxIndex(kp_day_night)][0]:.2f}')
ax[1].pcolormesh(vsys, op_post_night, cc_night_post_night)
ax[1].plot(K[maxIndex(kp_night_night)][0] * np.sin(2 * np.pi * op_post_night), op_post_night, color='red', ls='--', label = '$K_p$ = ' + f'{ K[maxIndex(kp_night_night)][0]:.2f}')
ax[0].legend(loc ='upper left')
ax[1].legend(loc ='upper left')

plt.savefig(os.path.join(local_images, "cc_day-night_post_night.png"))


plt.show()


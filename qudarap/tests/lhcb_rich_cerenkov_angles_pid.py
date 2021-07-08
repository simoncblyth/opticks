#!/usr/bin/env python
"""
lhcb_rich_cerenkov_angles_pid.py 
===================================


https://physics.stackexchange.com/questions/471722/cherenkov-light-and-refractive-index


"""
import numpy as np, os
import matplotlib.pyplot as plt

def beta(E, mec2):
    gamma = E / mec2
    return np.sqrt(1 - 1 / np.power(gamma, 2))

def theta(beta, n):
    return np.arccos(1 / (beta * n))

mec2_kaon = 493.68  # MeV / c^2
mec2_pion = 139.57  # MeV / c^2
mec2_muon = 105.65  # MeV / c^2

E = np.logspace(np.log10(5e3), 6, 100) # MeV

beta_pion = beta(E, mec2_pion)
beta_kaon = beta(E, mec2_kaon)
beta_muon = beta(E, mec2_muon)

n_1 = 1.0014
n_2 = 1.0005

fig, ax = plt.subplots()

plt.semilogx(E, theta(beta_kaon, n_1), color="crimson", ls="-", label=r"$K, n_1=1.0014$")
plt.semilogx(E, theta(beta_pion, n_1), color="crimson", ls="--", label=r"$\pi, n_1=1.0014$")
plt.semilogx(E, theta(beta_muon, n_1), color="crimson", ls="-.", label=r"$\mu, n_1=1.0014$")

plt.semilogx(E, theta(beta_kaon, n_2), color="k", ls="-", label=r"$K, n_2=1.0005$")
plt.semilogx(E, theta(beta_pion, n_2), color="k", ls="--", label=r"$\pi, n_2=1.0005$")
plt.semilogx(E, theta(beta_muon, n_2), color="k", ls="-.", label=r"$\mu, n_2=1.0005$")

plt.axvline(8e4, color="crimson", ls=":")
plt.axvline(1.1e5, color="k", ls=":")

plt.xlabel("E / MeV")
plt.ylabel(r"$\theta_{\rm C}\,/\,rad$")
plt.legend()
plt.show()

path = "/tmp/qudarap/lhcb_rich_cerenkov_angles_pid.png"
fold = os.path.dirname(path)
if not os.path.isdir(fold):
    os.mkdirs(fold)
pass
fig.savefig(path)


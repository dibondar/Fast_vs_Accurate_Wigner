__doc__ = """
To check accuracy of FastWigner vs AccurateWigner propagator
using the soft core Coulomb potential as an example
"""

import matplotlib.pyplot as plt

# Import propagators
from FastWigner.wigner_bloch_cuda_1d import WignerBlochCUDA1D as FastWignerPropagator
from FastWigner.wigner_normalize import WignerNormalize, WignerSymLogNorm
from AccurateWigner.wigner_bloch_cuda_1d import WignerBlochCUDA1D as AccurateWignerPropagator

##########################################################################################
#
#   Parameters of quantum systems
#
##########################################################################################
sys_params = dict(
    t=0.,
    dt=0.01,

    X_gridDIM=1024,
   # X_amplitude=30.,

    P_gridDIM=1024,
    #P_amplitude=40.,

    # the kinetic energy
    K="0.5 * P * P",

    # the soft core Coulomb potential for Ar
    V="-1. / sqrt(X * X + 1.37)",
)

##########################################################################################
#
#   Propagate
#
##########################################################################################

import numpy as np
acc_prop = AccurateWignerPropagator(X_amplitude=30.,  P_amplitude=60., **sys_params)
acc_ground_state = acc_prop.get_ground_state(abs_tol_purity=1e-10).get().real
acc_propagated_state = acc_prop.propagate(10000).get().real

fast_prop = FastWignerPropagator(X_amplitude=30., P_amplitude=20.,**sys_params)
fast_ground_state = fast_prop.get_ground_state(abs_tol_purity=1e-10).get()
fast_propagated_state = fast_prop.propagate(10000).get()

##########################################################################################
#
#   Plotting Fast Propagator
#
##########################################################################################

# plotting params
img_params = dict(
    extent=[fast_prop.X.min(), fast_prop.X.max(), fast_prop.P.min(), fast_prop.P.max()],
    origin='lower',
    aspect=1,
    cmap='bwr',
    norm=WignerNormalize(vmin=-1e-7, vmax=1e-7),
)

plt.subplot(221)
#plt.title("Ground state Wigner function obtained via Fast Wigner propagator")
plt.text(-28, -8, '(a)', color='k', fontsize=15)
plt.imshow(fast_ground_state, **img_params)
#plt.xlim([-20., 20])
plt.ylim([-10., 10])
plt.xlabel('$x$ (a.u.)')
plt.ylabel('$p$ (a.u.)')

plt.subplot(222)
#plt.title("Propagated state via Fast Wigner propagator")
plt.text(-28, -8, '(b)', color='k', fontsize=15)
plt.imshow(fast_propagated_state, **img_params)
#plt.xlim([-20., 20])
plt.ylim([-10., 10])
plt.xlabel('$x$ (a.u.)')
#plt.ylabel('$p$ (a.u.)')

##########################################################################################
#
#   Plotting Accurate Propagator
#
##########################################################################################

# just change the grid parameters
img_params["extent"]=[acc_prop.X.min(), acc_prop.X.max(), acc_prop.P.min(), acc_prop.P.max()]

plt.subplot(223)
#plt.title("Ground state Wigner function obtained via Accurate Wigner propagator")
plt.text(-28, -8, '(c)', color='k', fontsize=15)
plt.imshow(acc_ground_state, **img_params)
#plt.xlim([-20., 20])
plt.ylim([-10., 10])
plt.xlabel('$x$ (a.u.)')
plt.ylabel('$p$ (a.u.)')

plt.subplot(224)
#plt.title("Propagated state via Accurate Wigner propagator")
plt.text(-28, -8, '(d)', color='k', fontsize=15)
plt.imshow(acc_propagated_state, **img_params)
#plt.xlim([-40., 40])
plt.ylim([-10., 10])
plt.xlabel('$x$ (a.u.)')
#plt.ylabel('$p$ (a.u.)')

plt.show()



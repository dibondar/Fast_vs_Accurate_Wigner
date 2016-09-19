__doc__ = """
To check accuracy of FastWigner vs AccurateWigner propagator
using the Morese potential as an example
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
    X_amplitude=20.,

    P_gridDIM=1024,
    # P_amplitude=20.,

    # the kinetic energy
    K="0.5 * P * P",

    # the Morse potential from Fig. 1 of http://dx.doi.org/10.1103/PhysRevLett.114.050401
    V="15. * (exp(-0.5 * (X + 7.)) - 2. * exp(-0.25 * (X + 7.)))",

    #functions="""
    #// Blackman filter to be used as absorbing boundary condition
    #__device__ double blackman2D(const int i, const int j)
    #{
    #    const double Cj = j * 2.0 * M_PI / (X_gridDIM - 1.);
    #    const double Ci = i * 2.0 * M_PI / (P_gridDIM - 1.);
    #
    #    return  (0.42 - 0.5 * cos(Cj) + 0.08 * cos(2.0 * Cj)) *
    #            (0.42 - 0.5 * cos(Ci) + 0.08 * cos(2.0 * Ci));
    #}
    #""",

    #alpha=0.01,

    # define absorbing boundary for FastWignerPropagator
    #abs_boundary_x_p="pow(abs(blackman2D(i, j)), alpha * dt)",

    # define absorbing boundary for AccurateWignerPropagator
    #abs_boundary_x="pow(abs(blackman2D(i, j)), alpha * dt)",
)

##########################################################################################
#
#   Propagate
#
##########################################################################################

acc_prop = AccurateWignerPropagator(P_amplitude=30., **sys_params)
acc_ground_state = acc_prop.get_ground_state(abs_tol_purity=1e-12).get().real
acc_propagated_state = acc_prop.propagate(10000).get().real

fast_prop = FastWignerPropagator(P_amplitude=10., **sys_params)
fast_ground_state = fast_prop.get_ground_state(abs_tol_purity=1e-12).get()
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
    norm=WignerNormalize(vmin=-5e-10, vmax=5e-10),
)

plt.subplot(221)
#plt.title("Ground state Wigner function obtained via Fast Wigner propagator")
plt.text(-13, -8, '(a)', color='k', fontsize=15)
plt.imshow(fast_ground_state, **img_params)
plt.xlim([-15., 20])
#plt.ylim([-10., 10])
plt.xlabel('$x$ (a.u.)')
plt.ylabel('$p$ (a.u.)')

plt.subplot(222)
#plt.title("Propagated state via Fast Wigner propagator")
plt.text(-13, -8, '(b)', color='k', fontsize=15)
plt.imshow(fast_propagated_state, **img_params)
plt.xlim([-15., 20])
#plt.ylim([-10., 10])
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
plt.text(-13, -8, '(c)', color='k', fontsize=15)
plt.imshow(acc_ground_state, **img_params)
plt.xlim([-15., 20])
plt.ylim([-10., 10])
plt.xlabel('$x$ (a.u.)')
plt.ylabel('$p$ (a.u.)')

plt.subplot(224)
#plt.title("Propagated state via Accurate Wigner propagator")
plt.text(-13, -8, '(d)', color='k', fontsize=15)
plt.imshow(acc_propagated_state, **img_params)
plt.xlim([-15., 20])
plt.ylim([-10., 10])
plt.xlabel('$x$ (a.u.)')
#plt.ylabel('$p$ (a.u.)')

plt.show()
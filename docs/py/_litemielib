from numba import njit, int32, float64, complex128
from numpy import abs as npabs
from numpy import sum as npsum
from numpy import exp as npexp
from numpy import complex128 as npcomplex128
from numpy import float64 as npfloat64
from numpy import sin, cos, isscalar, arange, zeros, array

''' Adapted from the Miepython module to match the notation used in ADDA.
    See <https://miepython.readthedocs.io> for the complete original code. '''


''' index '''

    ###  m  = complex refractive index, imaginary part should be
    ###  x  = size parameter of the sphere, radius times the wavenumber k
    ###  mu = cos(theta), may be scalar or array

    ###  g = asymmetry parameter
    ###  Qext  = extinction efficiency
    ###  Qsca  = scattering efficiency
    ###  Qabs  = absorption efficiency
    ###  Qbck  = backscatter efficiency
    ###  S1,S2 = diagonal elements of the complex scattering amplitude

    ###  The code uses the Wiscombe normalisation for the scattering amplitude
    ###  its square modulus is the differential scattering cross-section, times
    ###  k**2. Units are sr**(-0.5) and the integral over 4pi is Qsca*pi*x**2
    ###  The number of coefficients is estimated based on Wiscombe's formula
    ###  chosen so that the error when the series are summed is ~ 1e-6
    ###  Applied Optics, Vol. 19, No. 9 1980

__all__ = ( 'mie_Q', 'mie_S', 'mie_S_')



### Riccati-Bessel ##############################################################

@njit((complex128, int32), cache=True)
def _Lentz_Dn(z, N):

    ''' Compute the logarithmic derivative of the Riccati-Bessel function '''

    ### Args:
    ###  z: function argument
    ###  N: order of Riccati-Bessel function

    ### Returns:
    ###  Riccati-Bessel function of order N with argument z using the
    ###  continued fraction technique of Lentz, Appl. Opt., 15, 668-671, (1976).

    zinv = 2.0 / z    ### there's a 2.0 for a reason
    alpha = (N + 0.5) * zinv
    aj = -(N + 1.5) * zinv
    alpha_j1 = aj + 1.0 / alpha
    alpha_j2 = aj
    ratio = alpha_j1 / alpha_j2
    runratio = alpha * ratio

    while npabs(npabs(ratio) - 1.0) > 1e-12:
        aj = zinv - aj
        alpha_j1 = aj + 1.0 / alpha_j1
        alpha_j2 = aj + 1.0 / alpha_j2
        ratio = alpha_j1 / alpha_j2
        zinv *= -1.0
        runratio = ratio * runratio

    return -N / z + runratio


@njit((complex128, int32, complex128[:]), cache=True)
def _D_downwards(z, N, D):

    ''' Compute the logarithmic derivative by downwards recurrence '''

    ### Args:
    ###  z: function argument
    ###  N: order of Riccati-Bessel function
    ###  D: gets filled with the Riccati-Bessel function values for orders
    ###     from 0 to N with argument z using the downwards recurrence relations.

    last_D = _Lentz_Dn(z, N)
    for n in range(N, 0, -1):
        n_z = n / z
        last_D = n_z - 1.0 / (last_D + n_z)
        D[n-1] = last_D


@njit((complex128, int32, complex128[:]), cache=True)
def _D_upwards(z, N, D):

    ''' Compute the logarithmic derivative by upwards recurrence '''

    ### Args:
    ###  z: function argument
    ###  N: order of Riccati-Bessel function
    ###  D: gets filled with the Riccati-Bessel function values for orders
    ###     from 0 to N with argument z using the upwards recurrence relations.

    exp = npexp(-2.0j * z)
    zinv = 1.0 / z
    D[1] = - zinv + (1.0 - exp) / ( (1.0 - exp)*zinv - 1.0j*(1.0 + exp) )
    for n in range(2, N):
        n_z = n * zinv
        D[n] = 1.0 / (n_z - D[n-1]) - n_z


@njit((complex128, float64, int32), cache=True)
def _D_calc(m, x, N):

    ''' Compute the logarithmic derivative using best method '''

    ### Args:
    ###  m: the complex index of refraction of the sphere
    ###  x: the size parameter of the sphere
    ###  N: order of Riccati-Bessel function

    ### Returns:
    ###  The values of the Riccati-Bessel function for orders from 0 to N.

    n = m.real
    kappa = npabs(m.imag)
    D = zeros(N, dtype=npcomplex128)

    if n < 1 or n > 10 or kappa > 10.0 or x * kappa >= 3.9 - 10.8 * n + 13.78 * n*n:
        _D_downwards(m*x, N, D)
    else:
        _D_upwards(m*x, N, D)
    return D


### Calc Mie coefficients #######################################################

@njit((complex128, float64), cache=True)
def _mie_an_bn(m, x):

    ''' Compute arrays of Mie coefficients a_n and b_n for a sphere '''

    ### Args:
    ###  m: the complex index of refraction of the sphere
    ###  x: the size parameter of the sphere

    nmax = int(1.01*x + 4.2 * x**0.333334 + 3.0)
    a_ = zeros(nmax - 1, dtype=npcomplex128)
    b_ = zeros(nmax - 1, dtype=npcomplex128)

    cosx = cos(x)
    psi_nm1 = sin(x)                 ### nm1 = n-1 = 0
    psi_n = psi_nm1 / x - cosx       ### n = 1
    xi_nm1 = complex(psi_nm1,cosx)
    xi_n = complex(psi_n,cosx / x + psi_nm1)

    if m.real > 0.0:
        D = _D_calc(m, x, nmax + 1)

        for n in range(1, nmax):
            temp = D[n] / m + n / x
            a_[n-1] = (temp * psi_n - psi_nm1) / (temp * xi_n - xi_nm1)
            temp = D[n] * m + n / x
            b_[n-1] = (temp * psi_n - psi_nm1) / (temp * xi_n - xi_nm1)
            xi = (2*n + 1) * xi_n / x - xi_nm1
            xi_nm1 = xi_n
            xi_n = xi
            psi_nm1 = psi_n
            psi_n = xi_n.real

    else:
        for n in range(1, nmax):
            a_[n-1] = (n * psi_n / x - psi_nm1) / (n * xi_n / x - xi_nm1)
            b_[n-1] = psi_n / xi_n
            xi = (2*n + 1) * xi_n / x - xi_nm1
            xi_nm1 = xi_n
            xi_n = xi
            psi_nm1 = psi_n
            psi_n = xi_n.real

    return nmax - 1, a_.conj(), b_.conj()


### Calc scattering coefficients ################################################

@njit((complex128, float64), cache=True)
def _small_conducting_mie(m,x):

    ''' Compute the efficiencies for a small conducting sphere (x<0.1, m.real=0) '''

    ### Args:
    ###  m: complex refractive index
    ###  x: size parameter = k * radius of the sphere

    x2 = x**2
    x3 = x*x2
    x4 = x2*x2
    ahat1 = complex(0.0, 2.0/3.0 * (1.0 - 0.2 * x2))
    ahat1 /= complex(1.0 - 0.5*x2, 2.0/3.0 * x3)

    bhat1 = complex(0.0, (x2 - 10.0)/30.0)
    bhat1 /= complex(1.0 + 0.5*x2, -x3/3.0)
    ahat2 = complex(0.0, x2/30.0)
    bhat2 = complex(0.0, -x2/45.0)

    Qsca = x4 * (  6.0 * npabs(ahat1)**2
                 + 6.0 * npabs(bhat1)**2
                 + 10.0 * npabs(ahat2)**2
                 + 10.0 * npabs(bhat2)**2 )
    Qext = Qsca
    g = ahat1.imag * (ahat2.imag + bhat1.imag)
    g += bhat2.imag * (5.0/9.0 * ahat2.imag + bhat1.imag)
    g += ahat1.real * bhat1.real
    g *= 6.0 * x4/Qsca

    Qbck = 9.0 * x4 * npabs( ahat1 - bhat1 - 5.0/3.0 * (ahat2 - bhat2) )**2

    return [Qext, Qsca, 0.0, Qbck, g]


@njit((complex128, float64), cache=True)
def _small_mie(m, x):

    ''' Calculate the efficiencies for a small sphere (x<0.1) '''

    ### Args:
    ###  m: complex refractive index
    ###  x: size parameter = k * radius of the sphere

    m2 = m*m
    x2 = x*x
    x3 = x*x2
    x4 = x2*x2

    D = m2 + 2.0 + (1.0 - 0.7 * m2) * x2
    D -= (8.0 * m**4 - 385.0 * m2 + 350.0) * x4 / 1400.0
    D += 2.0j * (m2 - 1.0) * x3 * (1.0 - 0.1 * x2) / 3.0
    ahat1 = 2.0j * (m2 - 1.0)/3.0 * (1.0 - 0.1*x2 + (4.0*m2 + 5.0) * x4 / 1400.0) / D

    bhat1 = 1.0j * x2 * (m2 - 1.0) / 45.0 * (1.0 + (2.0 * m2 - 5.0) / 70.0 * x2)
    bhat1 /= 1.0 - (2.0 * m2 - 5.0) / 30.0 * x2

    ahat2 = 1.0j * x2 * (m2 - 1.0) / 15.0 * (1.0 - x2 / 14.0)
    ahat2 /= 2.0*m2 + 3.0 - (2.0*m2 - 7.0) / 14.0 * x2

    T = npabs(ahat1)**2 + npabs(bhat1)**2 + 5.0/3.0 * npabs(ahat2)**2
    temp = ahat2 + bhat1
    g = (ahat1 * temp.conjugate()).real / T

    Qsca = 6 * x4 * T

    if m.imag == 0:
        Qext = Qsca
    else:
        Qext = 6.0*x * (ahat1 + bhat1 + 5.0*ahat2 / 3.0).real

    sbck = 1.5 * x3 * (ahat1 - bhat1 - 5.0*ahat2 / 3.0)
    Qbck = 4.0 * npabs(sbck)**2 / x2

    return [Qext, Qsca, Qext-Qsca, Qbck, g]


@njit((complex128, float64), cache=True)
def _mie_Q_scalar(m, x):

    ''' Calculate the efficiencies for a sphere when both m and x are scalars '''

    ### Args:
    ###  m: complex refractive index
    ###  x: size parameter = k * radius of the sphere

    if m.real == 0 and x < 0.1:
        Qext, Qsca, Qbck, g = _small_conducting_mie(m,x)

    elif m.real > 0.0 and npabs(m) * x < 0.1:
        Qext, Qsca, Qbck, g = _small_mie(m, x)

    else:
        nmax, a_, b_ = _mie_an_bn(m, x)

        x2 = x*x
        n = arange(1, nmax + 1)
        cn = 2.0*n + 1.0

        Qext = 2.0 * npsum(cn * (a_.real + b_.real)) / x2

        Qsca = Qext

        if m.imag != 0:
            Qsca = 2.0 * npsum(cn * (npabs(a_)**2 + npabs(b_)**2)) / x2

        Qbck = npabs(npsum((-1)**n * cn * (a_ - b_)))**2 / x2

        c1n = n * (n + 2) / (n + 1)
        c2n = cn / n / (n + 1)
        g = 0.0

        for i in range(nmax - 1):
            asy1 = c1n[i] * (a_[i] * a_[i + 1].conjugate() + b_[i] * b_[i + 1].conjugate()).real
            asy2 = c2n[i] * (a_[i] * b_[i].conjugate()).real
            g += 4.0 * (asy1 + asy2) / Qsca / x2

    return [Qext, Qsca, Qext-Qsca, Qbck, g]


@njit((complex128, float64, float64), cache=True)
def _mie_S_scalar(m, x, mu):

    ''' Compute the elements S1, S2 of the scattering amplitude (scalar) '''

    ### Args:
    ###  m: complex refractive index
    ###  x: size parameter of the sphere
    ###  mu: cos(theta)

    nmax, a_, b_ = _mie_an_bn(m, x)

    S1 = 0
    S2 = 0
    pi_nm2 = 0.0
    pi_nm1 = 1.0

    for n in range(1, nmax):
        np1 = n + 1
        twonp1 = n + np1  ### (2*n + 1)
        tau_nm1 = n * mu * pi_nm1 - np1 * pi_nm2

        S1 += twonp1 * (pi_nm1 * a_[n-1] + tau_nm1 * b_[n-1]) / (np1*n)
        S2 += twonp1 * (tau_nm1 * a_[n-1] + pi_nm1 * b_[n-1]) / (np1*n)

        temp = pi_nm1
        pi_nm1 = (twonp1 * mu * pi_nm1 - np1 * pi_nm2) / n
        pi_nm2 = temp

    return [S1, S2]


''' public methods '''###########################################################


def mie_Q(m_, x_):

    ''' Calculate the efficiencies for a sphere, m_ or x_ may be arrays '''

    ### Args:
    ###  m_: complex refractive index (may be array)
    ###  x_: size parameter of the sphere (may be array)

    mm = m_
    xx = x_

    if isscalar(m_) and isscalar(x_):
        return _mie_Q_scalar(mm, xx)

    if isscalar(m_): mlen = 0
    else: mlen = len(m_)

    if isscalar(x_): xlen = 0
    else: xlen = len(x_)

    if xlen > 0 and mlen > 0 and xlen != mlen:
        raise RuntimeError('m_ and x_ arrays must have same length')

    thelen = max(xlen, mlen)
    Qext = zeros(thelen, dtype=npfloat64)   ### the loop will then fill elements
    Qsca = zeros(thelen, dtype=npfloat64)
    Qabs = zeros(thelen, dtype=npfloat64)
    Qbck = zeros(thelen, dtype=npfloat64)
    g    = zeros(thelen, dtype=npfloat64)

    for i in range(thelen):

        if mlen > 0: mm = m_[i]
        if xlen > 0: xx = x_[i]

        Qext[i], Qsca[i], Qabs[i], Qbck[i], g[i] = _mie_Q_scalar(mm, xx)

    return [Qext, Qsca, Qabs, Qbck, g]

def mie_S(m, x, mu_):

    ''' Compute the elements S1, S2 of the scattering amplitude (array) '''

    ### Args:
    ###  m: complex refractive index
    ###  x: size parameter of the sphere
    ###  mu_: array of cos(theta), handle the scalar case with the first line

    if isscalar(mu_):
        return _mie_S_scalar(m, x, mu_)
    else:
        S_ = array([_mie_S_scalar(m, x, mu) for mu in mu_])
        return [S_[:,0], S_[:,1]]

@njit((complex128, float64, float64[:]), cache=True)
def mie_S_(m, x, mu_):

    ''' Compute the elements S1, S2 of the scattering amplitude (array) '''

    ### Args:
    ###  m: complex refractive index
    ###  x: size parameter of the sphere
    ###  mu_: array of cos(theta), handle the scalar case with the first line

    nmax, a_, b_ = _mie_an_bn(m, x)

    nangles = len(mu_)
    S1_ = zeros(nangles, dtype=npcomplex128)
    S2_ = zeros(nangles, dtype=npcomplex128)

    for k in range(nangles):
        pi_nm2 = 0.0
        pi_nm1 = 1.0
        for n in range(1, nmax):
            np1 = n + 1
            twonp1 = n + np1  ### (2*n + 1)
            tau_nm1 = n * mu_[k] * pi_nm1 - np1 * pi_nm2

            S1_[k] += twonp1 * (pi_nm1 * a_[n-1] + tau_nm1 * b_[n-1]) / (np1*n)
            S2_[k] += twonp1 * (tau_nm1 * a_[n-1] + pi_nm1 * b_[n-1]) / (np1*n)

            temp = pi_nm1
            pi_nm1 = (twonp1 * mu_[k] * pi_nm1 - np1 * pi_nm2) / n
            pi_nm2 = temp

    return [S1_, S2_]

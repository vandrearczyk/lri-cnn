"""
Spherical Harmonics convolution core implementation
"""

import numpy as np
import tensorflow as tf
import math
from math import pi
from scipy import special as sp
from sympy.physics.quantum.spin import Rotation
from sympy.physics.quantum.cg import CG


def sh_conv3d(X, out_ch, ksize, strides=(1, 1, 1, 1, 1), padding='VALID', degreeMax=1, stddev=0.4, name='shconv3d',
              is_trainable=True, is_hn=False):
    """
        3D Spherical Convolution used in S-LRI and SSE-LRI
        returns responses to SHs hYnm
        Parameters
        ----------
        X : tf tensor
            shape [batch, in_depth, in_height, in_width, 2 (complex), in_channels]
        out_ch : tf tensor
            shape [filter_radius, in_channels, out_channels], the radial profiles of the spherical harmonic filters
        ksize : int
            size of the kernel
        strides : tuple of int, optional
            size of the stride
        padding : string
            size of the padding
        degreeMax : int, optional
            Maximal degree in the spherical harmonics expansion
        stddev : float, optional
            stdde for the weight initialisation
        name : string, optional
            name for the tf graph
        is_trainable : bool, optional
            if True optimize the paramters of the layer
        is_hn : bool, optional
            if True, compute a different radial profile for each degree in the spherical harmonic expansion

        Returns
        -------
        _ : tf tensor
            The tensor output of this layer with shape

        """

    xsh = X.get_shape().as_list()
    in_ch = xsh[4]
    H = get_radials_dict(ksize, in_ch, out_ch, degreeMax, is_hn, is_trainable, std_mult=stddev, name='H' + name)
    hn = get_radial_volume(H, ksize, is_hn)
    Y = get_harmonics(degreeMax, ksize)
    if is_hn:
        hY = []
        for n in range(degreeMax + 1):
            hY.append(tf.multiply(tf.expand_dims(hn[:, :, :, n, :, :], -1),
                                  tf.expand_dims(tf.expand_dims(Y[..., n ** 2:(n + 1) ** 2], 3), 3)))
        hY = tf.concat(hY, -1)
    else:
        hY = tf.multiply(tf.expand_dims(hn, -1), tf.expand_dims(tf.expand_dims(Y, 3), 3))
    # Convolve to all filters then add (first reshape hY in order to convolve all at once)
    R = conv3d_complex(X, tf.reshape(hY, [ksize, ksize, ksize, in_ch, out_ch * (degreeMax + 1) ** 2]), strides=strides,
                       padding=padding, name=name)

    return tf.reshape(R, [tf.shape(R)[0], R.shape[1], R.shape[2], R.shape[3], out_ch, (degreeMax + 1) ** 2])


def conv3d_complex(X, W, strides=(1, 1, 1, 1), padding='VALID', name='conv3d_complex'):
    """
    Returns the 3D convolution of the inputs X with the complex filters W
    Parameters
    ----------
    X : tf tensor
        input of the layer with shape (bs,h,w,d,in_ch)
    W : tf tensor
        filters complex tensors of shape (h,w,d,in_ch,out_ch)

    Returns
    -------
    complex tf tensor,
        complex tensor shape (bs,h,w,d,out_ch)
    """

    Wsh = W.get_shape().as_list()
    # Reshape filters
    W_ = tf.stack([tf.math.real(W), tf.math.imag(W)], 4)
    W_ = tf.reshape(W_, (Wsh[0], Wsh[1], Wsh[2], Wsh[3], 2 * Wsh[4]))
    # Convolve
    Y = tf.nn.conv3d(X, W_, strides=strides, padding=padding, name=name)
    # Reshape result into appropriate shape
    Ysh = Y.get_shape().as_list()
    Y = tf.reshape(Y, (tf.shape(Y)[0], Ysh[1], Ysh[2], Ysh[3], 2, int(Ysh[4] / 2)))

    return tf.complex(Y[:, :, :, :, 0, :], Y[:, :, :, :, 1, :])


def s_conv3d(X, out_ch, ksize, strides=(1, 1, 1, 1, 1), padding='VALID', degreeMax=1, stddev=0.4, name='conv3d', M=4,
             is_trainable=True, is_hn=False):
    """
        Returns the S-LRI convolution responses after orientation pooling.
        Parameters
        ----------
        X : tf tensor
            input tensor of shape (bs,w,h,d,in_ch)
        out_ch : int
            number of output channels
        ksize : int
            kernel size
        degreeMax : int, optional
            maximal SH degree
        M : int, optional
            number of orientation sampled on the sphere
        is_trainable
        is_hn : bool, optional
             if True, compute a different radial profile for each degree in the spherical harmonic expansion

        Returns
        -------
        _ :
    """

    in_ch = X.get_shape().as_list()[4]
    # Coefficients Cnm
    C = get_shcoeffs_dict(in_ch, out_ch, degreeMax, is_trainable, std_mult=stddev, name='C' + name)
    Sr = get_steering_matrix(degreeMax, ksize, M)
    R = sh_conv3d(X, out_ch, ksize, strides, padding, degreeMax, stddev, name, is_trainable, is_hn)
    return steer(R, C, Sr, degreeMax)


def sse_conv3d(X, out_ch, ksize, strides=(1, 1, 1, 1, 1), padding='VALID', degreeMax=1, stddev=0.4, name='conv3d', M=4,
               is_trainable=True, is_hn=False):
    """

    Parameters
    ----------
    X : tf tensor
        input tensor of shape (bs,w,h,d,in_ch)
     out_ch : int
        number of output channels
    ksize : int
        kernel size
    degreeMax : int, optional
        maximal SH degree
    M : int, optional
        number of orientation sampled on the sphere
    is_hn : bool, optional
         if True, compute a different radial profile for each degree in the spherical harmonic expansion

    Returns
    -------
    _ : tf tensor
        the Solid Spherical Energy (SSE)-LRI convolution responses.
    """
    # Create a matrix for quickly summing the norms within degrees n
    mat = np.zeros(((degreeMax + 1) ** 2, degreeMax + 1))
    for n in range(degreeMax + 1):
        mat[n ** 2:(n + 1) ** 2, n] = 1
    mat = tf.constant(mat, dtype=tf.float32)
    R = sh_conv3d(X, out_ch, ksize, strides, padding, degreeMax, stddev, name, is_trainable, is_hn)
    return get_sse_invariant_coeffs(R, degreeMax, mat)


def get_variables(filter_shape, is_trainable=True, W_init=None, std_mult=0.4,
                  name='R'):  # this works for both radials and weights.
    """
    Initialize variable with He method
    filter_shape: list of filter dimensions
    W_init: numpy initial values (default None)
    std_mult: multiplier for weight standard deviation (default 0.4)
    name: (default W)
    """
    return tf.get_variable(name, dtype=tf.float32, shape=filter_shape, initializer=W_init, trainable=is_trainable)


def get_radial_volume(H, ksize, is_hn):
    """
    Returns the 3d volume from the 1d polar separable radial profile for each in_ch, out_ch.
    Output shape:
        (ksize,ksize,ksize,in_ch,out_ch)
        or if is_hn (ksize,ksize,ksize,degreeMax+1,in_ch,out_ch)
    H: input tensor of shape:
        (n_grid,out_ch)
        or
        if is_hn (n_grid, degreeMax+1,in_ch,out_ch)
    is_hn: Whether we have one radial profile per degree n (non-polar-separable)
    """
    if is_hn:
        n_grid, degreeMax, in_ch, out_ch = H.get_shape().as_list()
        degreeMax -= 1  # The shape has dimension degreeMax+1 because of degree 0.
        # Initialyze the volumes
        shape = [ksize, ksize, ksize, degreeMax + 1, in_ch, out_ch]
    else:
        n_grid, in_ch, out_ch = H.get_shape().as_list()
        # Initialyze the volumes
        shape = [ksize, ksize, ksize, in_ch, out_ch]
    radius = (ksize - 1) / 2
    radialVolume = tf.zeros(shape, dtype=tf.complex64)

    # Create the linear interpolation matrix y = |x-x1|y2+|x-x2|y1
    x = np.arange(-radius, radius + 1, 1)
    MG = np.ones((3, ksize, ksize, ksize))
    MG[0, :, :, :], MG[1, :, :, :], MG[2, :, :, :] = np.meshgrid(x, x, x)
    dist = np.reshape(np.sqrt(np.sum(np.square(MG), axis=0)), (ksize ** 3))  # distance to the center (flattened)
    distMat = np.tile(dist, (n_grid, 1))  # Replicate the distance for all n_grids
    a = np.arange(ksize ** 3)
    b = np.arange(n_grid)
    X = np.meshgrid(a, b)[1]
    M_interp = np.swapaxes((1 - np.abs(distMat - X)) * (np.floor(np.abs(distMat - X)) == 0).astype(np.float32), 0, 1)
    M_interp = tf.constant(M_interp.astype(np.float32))
    if is_hn:
        # reshape H for matrix mult
        H = tf.reshape(H, [n_grid, (degreeMax + 1) * in_ch * out_ch])
        # Do the matrix multiplication
        radialVolume = tf.reshape(tf.matmul(M_interp, H), [ksize, ksize, ksize, (degreeMax + 1), in_ch, out_ch])
        return tf.complex(radialVolume, tf.zeros((ksize, ksize, ksize, degreeMax + 1, in_ch, out_ch)))
    else:
        # reshape H for matrix mult
        H = tf.reshape(H, [n_grid, in_ch * out_ch])
        # Do the matrix multiplication
        radialVolume = tf.reshape(tf.matmul(M_interp, H), [ksize, ksize, ksize, in_ch, out_ch])
        return tf.complex(radialVolume, tf.zeros((ksize, ksize, ksize, in_ch, out_ch)))


def get_harmonics(degreeMax, ksize):
    """
    Returns the spherical harmonics for all degrees (n) and orders (m) specified by the maximum degree degreeMax
    Output: complex tensor of shape (ksize**3,(degreeMax+1)**2)
    degreeMax: maximal SH degree (degreeMax)
    ksize: kernel size
    """
    _, theta, phi = getSphCoordArrays(ksize)
    harmonics = []
    for n in range(degreeMax + 1):
        P_legendre = legendre(n, np.cos(theta))
        for m in range(-n, n + 1):
            # get the spherical harmonics (without radial profile)
            sh = spherical_harmonics(m, n, P_legendre, phi)
            # Reshape for multiplication with radial profile r
            harmonics.append(tf.constant(sh, dtype=tf.complex64))
    return tf.stack(harmonics, -1)


def get_steering_matrix(degreeMax, ksize, M):
    '''
    Returns a tensor of a block diagonal matrix Sr for the M orientations: i.e. a matrix of weights of shape ((degreeMax+1)**2,(degreeMax+1)**2) for each orientation.
    The output Sr has shape (M,(degreeMax+1)**2,(degreeMax+1)**2)).
    (degreeMax+1)**2 is the total number of Ynm, it comes from the sum_n(2n+1)
    degreeMax: maximal SH degree
    ksize: kernel size
    M: number of orientations
    '''
    radius = (ksize - 1) / 2
    radiusAug = int(np.ceil((radius - 1) * np.sqrt(3)) + 1)
    ksizeAug = radiusAug * 2 + 1
    # define search space for angles.
    zyz = get_euler_angles(M)
    _, theta, phi = getSphCoordArrays(ksize)

    Sr = np.zeros((M, (degreeMax + 1) ** 2, (degreeMax + 1) ** 2), dtype=np.complex64)
    # scan through angles
    for a in range(zyz.shape[0]):
        alpha, beta, gamma = zyz[a] * pi / 180
        # Import Wigner D matrix directly from simpy
        for n in range(degreeMax + 1):
            for k1 in range(n * 2 + 1):
                m1 = k1 - n
                for k2 in range(n * 2 + 1):
                    m2 = k2 - n
                    Sr[a, n ** 2 + k1, n ** 2 + k2] = np.complex(Rotation.D(n, m1, m2, alpha, beta, gamma).doit())

    return tf.constant(Sr)


def legendre(n, X):
    '''
    Legendre polynomial used to define the SHs for degree n
    '''
    res = np.zeros(((n + 1,) + (X.shape)))
    for m in range(n + 1):
        res[m] = sp.lpmv(m, n, X)
    return res


def spherical_harmonics(m, n, P_legendre, phi):
    '''
    Returns the SH of degree n, order m
    '''
    P_n_m = np.squeeze(P_legendre[np.abs(m)])
    sign = (-1) ** ((m + np.abs(m)) / 2)
    # Normalization constant
    A = sign * np.sqrt((2 * n + 1) / (4 * pi) * np.math.factorial(n - np.abs(m)) / np.math.factorial(n + np.abs(m)))
    # Spherical harmonics
    sh = A * np.exp(1j * m * phi) * P_n_m
    # Normalize the SH to unit norm
    sh /= np.sqrt(np.sum(sh * np.conj(sh)))
    return sh.astype(np.complex64)


def getSphCoordArrays(ksize):
    '''
    Returns spherical coordinates (rho,theta,phi) from the kernel size ksize
    '''
    if np.mod(ksize, 2):  # ksize odd
        x = np.linspace(-1, 1, ksize)
    else:  # ksize even
        x = np.linspace(-1, 1, ksize)  #
    MG = np.ones((3, ksize, ksize, ksize))
    X, Y, Z = np.meshgrid(x, x, x)
    rho = np.sqrt(np.sum(np.square([X, Y, Z]), axis=0))
    phi = np.squeeze(np.arctan2(Y, X))
    theta = np.nan_to_num(np.squeeze(np.arccos(Z / rho)))
    rho = np.squeeze(rho)
    return [rho, theta, phi]


def get_radials_dict(ksize, in_ch, out_ch, degreeMax, is_hn=True, is_trainable=True, std_mult=0.4, name='R'):
    '''
    Returns a tensor of radial profiles (trainable weights) of size (n_grid,degreeMax,in_ch,out_ch) if non-polar-separable (is_hn=True), or (n_grid,in_ch,out_ch) if polar-separable.
    ksize: kernel size
    in_ch: number of input channels
    out_ch: number of output channels
    degreeMax: maximal SH degree
    is_hn: Whether we have one radial profile per degree n (non-polar-separable)
    '''
    radius = int((ksize + 1) / 2)
    n_grid = int(np.ceil((radius - 1) * np.sqrt(3)) + 1)  # number of points for the radial profile
    if is_hn:
        shape = [n_grid, degreeMax + 1, in_ch, out_ch]  #
    else:
        shape = [n_grid, in_ch, out_ch]  #
    stddev = std_mult * np.sqrt(1.0)
    W_init = tf.random_normal_initializer(stddev=stddev)
    radials_dict = get_variables(shape, is_trainable, W_init, std_mult=std_mult, name=name)
    return radials_dict


def get_shcoeffs_dict(in_ch, out_ch, degreeMax, is_trainable=True, std_mult=0.4, name='W'):
    '''
    Returns a tensor of expansion coefficients (C) of shape (out_ch,(degreeMax+1)^2), where (degreeMax+1)^2 is the total number of SHs (Ynm)
    in_ch: number of input channels
    out_ch number of output channels
    degreeMax: maximal SH degree
    '''
    coeffs_dict = {}
    C = []
    stddev = std_mult * np.sqrt(2.0 / in_ch * (
                degreeMax + 1) ** 2)  # This sets the variance to 2/(in_ch*(degreeMax+1)^2), (degreeMax+1)^2 being the total number of SHs
    W_init = tf.random_normal_initializer(stddev=stddev)
    for n in range(degreeMax + 1):
        # First define trainable variables for the coefficients that need it (m>=0)
        for m in range(n + 1):  # Learn only half of the Cnm, the rest is calculated to obtain the symmetry
            name_nm = name + '_' + str(n) + str(m)
            if m == 0:  # The condition on C if m=0: Cn0=conj(Cn0). So Cn0 is real
                coeffs_dict[(n, m)] = get_variables([out_ch, ], is_trainable, W_init, std_mult=std_mult, name=name_nm)
            else:
                coeffs_dict[(n, m)] = get_variables([out_ch, 2], is_trainable, W_init, std_mult=std_mult,
                                                    name=name_nm)  # Now the coeffs are complex

        # Then fill a tensor with all Cnm values
        for k in range(n * 2 + 1):
            m = k - n
            if m == 0:
                C.append(tf.complex(coeffs_dict[(n, m)], 0.))
            elif m >= 0:
                C.append(tf.complex(coeffs_dict[(n, m)][..., 0], coeffs_dict[(n, m)][..., 1]))
            else:
                C.append(tf.scalar_mul((-1) ** (m), tf.math.conj(
                    tf.complex(coeffs_dict[(n, -m)][..., 0], coeffs_dict[(n, -m)][..., 1]))))
    return tf.stack(C, -1)


def get_sse_invariant_coeffs(R, degreeMax, mat):
    '''
    Returns the spectrum invariants from the responses to the SHs
    R: features maps responses to the SHs (Ynm)
    degreeMax: maximal SH degree (could be calculated from the shape of R)
    mat: block diagonal matrix for fast multiplication.
    '''
    bs = tf.shape(R)[0]
    rsh = R.get_shape().as_list()
    R = tf.reduce_sum(tf.square(tf.stack((tf.math.real(R), tf.math.imag(R)), 0)), 0)
    R = tf.matmul(tf.reshape(R, (bs * rsh[1] * rsh[2] * rsh[3] * rsh[4], (degreeMax + 1) ** 2)), mat)
    return tf.reshape(R, (bs, rsh[1], rsh[2], rsh[3], rsh[4] * (degreeMax + 1)))


def steer(R, C, Sr, degreeMax, Ms=None):
    '''
    Returns the steered responses for the set of steering matrices Sr, max pooled on the orientation channels. Output shape: (bs,h,w,d,out_ch,M)
    R: responses to the hYnm convolutions, complex tensor of shape (bs,h,w,d,out_ch,(degreeMax+1)^2)
    C: complex tensor of shape: (out_ch,(degreeMax+1)^2)
    Sr: block diagonal steering matrices of shape (M,(degreeMax+1)^2,(degreeMax+1)^2)
    degreeMax: maximal SH degree
    '''
    M = Sr.get_shape().as_list()[0]
    rsh = R.get_shape().as_list()
    bs = tf.shape(R)[0]
    Cr = tf.einsum('ij,ajk->aik', C, Sr)  # shape (M, out_ch, (degreeMax+1)^2)

    # Splitting real and imag
    R = tf.transpose(tf.reshape(R, [bs * rsh[1] * rsh[2] * rsh[3], rsh[4], rsh[5]]),
                     [1, 0, 2])  # shape (C, S, (degreeMax+1)^2), where C=bs.h.w.d
    Cr = tf.transpose(Cr, [1, 2, 0])  # shape (C, (degreeMax+1)^2, M)
    R = tf.stack((tf.math.real(R), tf.math.imag(R)), -1)  # shape (C, S, (degreeMax+1)^2, im)
    R = tf.reshape(R, (rsh[4], bs * rsh[1] * rsh[2] * rsh[3], rsh[5] * 2))  # shape (C, S, (degreeMax+1)^2*im)
    Cr = tf.stack((tf.math.real(Cr), -tf.math.imag(Cr)),
                  2)  # shape (C, (degreeMax+1)^2, im, M). There is a -tf.math.imag because i^2=-1
    Cr = tf.reshape(Cr, (rsh[4], rsh[5] * 2, M))  # shape (C, (degreeMax+1)^2*im, M)
    # loop over the channels out_ch
    Rs = tf.map_fn(lambda x: tf.reduce_max(tf.matmul(x[0], x[1]), 1), (R, Cr), dtype=tf.float32,
                   swap_memory=True)  # shape (out_ch, S)
    Rs = tf.reshape(Rs, (rsh[4], bs, rsh[1], rsh[2], rsh[3]))  # shape (out_ch,bs,h,w,d)
    return tf.transpose(Rs, [1, 2, 3, 4, 0])


def steer_M(R, C, Sr, degreeMax, Ms=None):
    '''
    Returns the steered responses for the set of steering matrices Sr. Output shape: (bs,h,w,d,out_ch*M)
    R: responses to the hYnm convolutions, complex tensor of shape (bs,h,w,d,out_ch,(degreeMax+1)^2)
    C: complex tensor of shape: (out_ch,(degreeMax+1)^2)
    Sr: block diagonal steering matrices of shape (M,(degreeMax+1)^2,(degreeMax+1)^2)
    degreeMax: maximal SH degree
    '''
    M = Sr.get_shape().as_list()[0]
    rsh = R.get_shape().as_list()
    bs = tf.shape(R)[0]
    Cr = tf.einsum('ij,ajk->aik', C, Sr)  # shape (M, out_ch, (degreeMax+1)^2)

    # Splitting real and imag
    R = tf.transpose(tf.reshape(R, [bs * rsh[1] * rsh[2] * rsh[3], rsh[4], rsh[5]]),
                     [1, 0, 2])  # shape (C, S, (degreeMax+1)^2), where C=bs.h.w.d
    Cr = tf.transpose(Cr, [1, 2, 0])  # shape (C, (degreeMax+1)^2, M)
    R = tf.stack((tf.math.real(R), tf.math.imag(R)), -1)  # shape (C, S, (degreeMax+1)^2, im)
    R = tf.reshape(R, (rsh[4], bs * rsh[1] * rsh[2] * rsh[3], rsh[5] * 2))  # shape (C, S, (degreeMax+1)^2*im)
    Cr = tf.stack((tf.math.real(Cr), -tf.math.imag(Cr)),
                  2)  # shape (C, (degreeMax+1)^2, im, M). There is a -tf.math.imag because i^2=-1
    Cr = tf.reshape(Cr, (rsh[4], rsh[5] * 2, M))  # shape (C, (degreeMax+1)^2*im, M)
    # loop over the channels out_ch
    Rs = tf.map_fn(lambda x: tf.matmul(x[0], x[1]), (R, Cr), dtype=tf.float32, swap_memory=True)  # shape (out_ch, S,M)
    # import pdb;pdb.set_trace()
    Rs = tf.transpose(Rs, [1, 0, 2])  # shape (S,out_ch,M)
    return tf.reshape(Rs, (bs, rsh[1], rsh[2], rsh[3], rsh[4] * M))  # shape (bs,h,w,d,out_ch*M)


def get_euler_angles(M):
    '''
    Returns the zyz Euler angles with shape (M, 3) for the defined number of orientations M.
    (intrinsic Euler angles in the zyz convention)
    '''
    if M == 1:
        zyz = np.array([[0, 0, 0]])
    elif M == 2:
        zyz = np.array([[0, 0, 0], [180, 0, 0]])
    elif M == 4:  # Implement Klein's four group see Worrall and Brostow 2018
        zyz = np.array([[0, 0, 0], [180, 0, 0], [0, 180, 0], [180, 180, 0]])
    elif M == 8:  # Test of theta and phi.
        zyz = np.array(
            [[0, 0, 0], [0, 45, 315], [0, 90, 270], [0, 135, 225], [0, 180, 180], [0, 225, 135], [0, 270, 90],
             [0, 315, 45]])
    elif M == 24:  # as represented in Worrall and Brostow 2018, derived from the Caley's table
        # For intrinsic Euler angles (each row is for one of the six points on the sphere (theta,phi angles))
        zyz = np.array([[0, 0, 0], [0, 0, 90], [0, 0, 180], [0, 0, 270],
                        [0, 90, 0], [0, 90, 90], [0, 90, 180], [0, 90, 270],
                        [0, 180, 0], [0, 180, 90], [0, 180, 180], [0, 180, 270],
                        [0, 270, 0], [0, 270, 90], [0, 270, 180], [0, 270, 270],
                        [90, 90, 0], [90, 90, 90], [90, 90, 180], [90, 90, 270],
                        [90, 270, 0], [90, 270, 90], [90, 270, 180], [90, 270, 270]
                        ])

    elif M == 72:  # as represented in Worrall and Brostow 2018, derived from the Caley's table
        # For intrinsic Euler angles (each row is for one of the six points on the sphere (theta,phi angles))
        zyz = np.array([[0, 0, 0], [0, 0, 90], [0, 0, 180], [0, 0, 270],
                        [0, 90, 0], [0, 90, 90], [0, 90, 180], [0, 90, 270],
                        [0, 180, 0], [0, 180, 90], [0, 180, 180], [0, 180, 270],
                        [0, 270, 0], [0, 270, 90], [0, 270, 180], [0, 270, 270],
                        [90, 90, 0], [90, 90, 90], [90, 90, 180], [90, 90, 270],
                        [90, 270, 0], [90, 270, 90], [90, 270, 180], [90, 270, 270],

                        [0, 45, 0], [0, 45, 90], [0, 45, 180], [0, 45, 270],
                        [0, 135, 0], [0, 135, 90], [0, 135, 180], [0, 135, 270],
                        [0, 225, 0], [0, 225, 90], [0, 225, 180], [0, 225, 270],
                        [0, 315, 0], [0, 315, 90], [0, 315, 180], [0, 315, 270],

                        [90, 45, 0], [90, 45, 90], [90, 45, 180], [90, 45, 270],
                        [90, 135, 0], [90, 135, 90], [90, 135, 180], [90, 135, 270],
                        [90, 225, 0], [90, 225, 90], [90, 225, 180], [90, 225, 270],
                        [90, 315, 0], [90, 315, 90], [90, 315, 180], [90, 315, 270],

                        [45, 90, 0], [45, 90, 90], [45, 90, 180], [45, 90, 270],
                        [135, 90, 0], [135, 90, 90], [135, 90, 180], [135, 90, 270],
                        [45, 270, 0], [45, 270, 90], [45, 270, 180], [45, 270, 270],
                        [135, 270, 0], [135, 270, 90], [135, 270, 180], [135, 270, 270]
                        ])

    else:  # TO DO
        raise ValueError("M = " + str(M) + " not yet implemented. Try 1, 4, 24 or 72")
        ''' XXX
        # Parametrized uniform triangulation of 3D circle/sphere:
        n_gamma = 4

        # No need for stlPoints AND A, B, C
        stlPoints, _, _, _ = sphereTriangulation(M,n_gamma)
        # Then do spherical coordinates to get the alpha and beta angles uniformly sampled on the sphere.
        alpha,beta = change_vars(np.swapaxes(stlPoints,0,1)) # The Euler angles alpha and beta are respectively theta and phi in spherical coord.
        # Then sample uniformly on gamma
        step_gamma = 2*pi/n_gamma
        gamma = np.tile(np.linspace(0,2*pi-step_gamma,n_gamma),alpha.shape[0])
        alpha = np.repeat(alpha,n_gamma)
        beta = np.repeat(beta,n_gamma)
        zyz = np.stack((alpha,beta,gamma),axis=1)*180.0/pi
        '''
    return zyz


def degreeToIndexes_range(n):
    return range(n * n, n * n + 2 * n + 1)


def degreeToIndexes_slice(n):
    return slice(n * n, n * n + 2 * n + 1)


def compute_cg_matrix(k, l):
    """
    Computes the matrix that block-diagonilizes the Kronecker product of Wigner D matrices of degree k and l respectively
    Output size (2k+1)(2l+1)x(2k+1)(2l+1)
    """
    c_kl = np.zeros([(2 * k + 1) * (2 * l + 1), (2 * k + 1) * (2 * l + 1)])

    n_off = 0
    for J in range(abs(k - l), k + l + 1):
        m_off = 0
        for m1_i in range(2 * k + 1):
            m1 = m1_i - k
            for m2_i in range(2 * l + 1):
                m2 = m2_i - l
                for n_i in range(2 * J + 1):
                    n = n_i - J
                    if m1 + m2 == n:
                        c_kl[m_off + m2_i, n_off + n_i] = CG(k, m1, l, m2, J, m1 + m2).doit()
            m_off = m_off + 2 * l + 1
        n_off = n_off + 2 * J + 1

    return c_kl


## NOT USED FOR NOW ##

def sphereTriangulation(M, n_gamma):
    """
    Defines points on the sphere that we use for alpha (z) and beta (y') Euler angles sampling. We can have 24 points (numIterations=0), 72 (numIterations=1), 384 (numIterations=2) etc.
    Copied from the matlab function https://ch.mathworks.com/matlabcentral/fileexchange/38909-parametrized-uniform-triangulation-of-3d-circle-sphere
    M is the number total of orientation, i.e. number of points on the sphere + number of angles for the gamma angle (n_gamma).

    """
    #
    numIter = int((M / 24) ** (1 / n_gamma) - 1)
    # function returns stlPoints fromat and ABC format if its needed,if not - just delete it and adapt to your needs
    radius = 1
    # basic Octahedron reff:http://en.wikipedia.org/wiki/Octahedron
    # ( ?1, 0, 0 )
    # ( 0, ?1, 0 )
    # ( 0, 0, ?1 )
    A = np.asarray([1, 0, 0]) * radius
    B = np.asarray([0, 1, 0]) * radius
    C = np.asarray([0, 0, 1]) * radius
    # from +-ABC create initial triangles which define oxahedron
    triangles = np.asarray([A, B, C,
                            A, B, -C,
                            # -x, +y, +-Z quadrant
                            -A, B, C,
                            -A, B, -C,
                            # -x, -y, +-Z quadrant
                            -A, -B, C,
                            -A, -B, -C,
                            # +x, -y, +-Z quadrant
                            A, -B, C,
                            A, -B, -C])  # -----STL-similar format
    # for simplicity lets break into ABC points...
    selector = np.arange(0, len(triangles[:, 1]) - 2, 3)
    Apoints = triangles[selector, :]
    Bpoints = triangles[selector + 1, :]
    Cpoints = triangles[selector + 2, :]
    # in every of numIterations
    for iteration in range(numIter):
        # devide every of triangle on three new
        #        ^ C
        #       / \
        # AC/2 /_4_\CB/2
        #     /\ 3 /\
        #    / 1\ /2 \
        # A /____V____\B           1st              2nd              3rd               4th
        #        AB/2
        # new triangleSteck is [ A AB/2 AC/2;     AB/2 B CB/2;     AC/2 AB/2 CB/2    AC/2 CB/2 C]
        AB_2 = (Apoints + Bpoints) / 2
        # do normalization of vector
        AB_2 = arsUnit(AB_2, radius)  # same for next 2 lines
        AC_2 = (Apoints + Cpoints) / 2
        AC_2 = arsUnit(AC_2, radius)
        CB_2 = (Cpoints + Bpoints) / 2
        CB_2 = arsUnit(CB_2, radius)
        Apoints = np.concatenate((Apoints,  # A point from 1st triangle
                                  AB_2,  # A point from 2nd triangle
                                  AC_2,  # A point from 3rd triangle
                                  AC_2))  # A point from 4th triangle..same for B and C
        Bpoints = np.concatenate((AB_2, Bpoints, AB_2, CB_2))
        Cpoints = np.concatenate((AC_2, CB_2, CB_2, Cpoints))
    # now tur points back to STL-like format....
    numPoints = np.shape(Apoints)[0]
    selector = np.arange(numPoints)
    selector = np.stack((selector, selector + numPoints, selector + 2 * numPoints))

    selector = np.swapaxes(selector, 0, 1)
    selector = np.concatenate(selector)
    stlPoints = np.concatenate((Apoints, Bpoints, Cpoints))
    stlPoints = stlPoints[selector, :]

    return stlPoints, Apoints, Bpoints, Cpoints


def change_vars(MG):
    """
    MG: np array of shape (3,...) containing 3D cartesian coordinates.
    returns spherical coordinates theta and phi (could return rho if needed)
    """
    rho = np.sqrt(np.sum(np.square(MG), axis=0))
    phi = np.squeeze(np.arctan2(MG[1, ...], MG[0, ...])) + pi
    theta = np.squeeze(np.arccos(MG[2, ...] / rho))
    # The center value is Nan due to the 0/0. So make it 0.
    theta[np.isnan(theta)] = 0
    rho = np.squeeze(rho)

    return theta, phi


def arsNorm(A):
    # vectorized norm() function
    rez = A[:, 0] ** 2 + A[:, 1] ** 2 + A[:, 2] ** 2
    rez = np.sqrt(rez)
    return rez


def arsUnit(A, radius):
    # vectorized unit() functon
    normOfA = arsNorm(A)
    rez = A / np.stack((normOfA, normOfA, normOfA), 1)
    rez = rez * radius
    return rez

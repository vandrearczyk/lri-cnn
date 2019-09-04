"""
Functions used for the different LRI networks
"""

import tensorflow as tf
import scipy
import numpy as np


########################## standard conv3d and GAP functions ##############################################

def conv3d(name, l_input, w, b, stride=1):
    return tf.nn.bias_add(
        tf.nn.conv3d(l_input, w, strides=[1, stride, stride, stride, 1], padding='VALID'),
        b
    )


def gavg_pool(name, X):
    """
    Returns the 3D global average pool of the feature maps, with or without orientation channel.
    Ouptut shape: (bs,out_ch) or (bs,out_ch*M)
    X: shape (bs,h,w,d,out_ch) or (bs,h,w,d,out_ch*M)
    """
    if len(X.get_shape().as_list()) != 5:
        raise ValueError('X unexpected shape. Expected shape (bs,h,w,d,out_ch) or (bs,h,w,d,out_ch*M).')
    return tf.reduce_mean(X, axis=[1, 2, 3])


########################## Functions to create dataset, pre-process and augment ##############################

def transform_matrix_offset_center_fixed(matrix, x, y, z):
    # Based on keras implementation that is wrong. It should be - 0.5 
    o_x = float(x) / 2 - 0.5
    o_y = float(y) / 2 - 0.5
    o_z = float(z) / 2 - 0.5
    offset_matrix = np.array([[1, 0, 0, o_x],
                              [0, 1, 0, o_y],
                              [0, 0, 1, o_z],
                              [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -o_x],
                             [0, 1, 0, -o_y],
                             [0, 0, 1, -o_z],
                             [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_affine_transform_fixed(x, theta_xyz=(0, 0, 0), tx=0, ty=0, tz=0, shear_xy=0, shear_xz=0, shear_yz=0,
                                 zx=1, zy=1, zz=1, row_axis=0, col_axis=1, z_axis=2,
                                 channel_axis=3, fill_mode='nearest', cval=0., order=1):
    """Applies an affine transformation specified by the parameters given.

    # Arguments
        x: 3D numpy array, single image.
        theta_xyz: rotation angles
        theta: Azimutal rotation angle in degrees.
        phi: Polar rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        tz: depth shift.
        shear_xy: Shear angle in degrees on the xy plane.
        shear_xz: Shear angle in degrees on the xz plane.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        zz: Zoom in z direction
        row_axis: Index of axis for rows in the input image.
        col_axis: Index of axis for columns in the input image.
        z_axis: Index of axis for depth in the input image.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """
    theta_x = theta_xyz[0]
    theta_y = theta_xyz[1]
    theta_z = theta_xyz[2]
    if scipy is None:
        raise ImportError('Image transformations require SciPy. '
                          'Install SciPy.')
    transform_matrix = None
    if theta_x != 0:
        theta = np.deg2rad(theta_x)
        rotation_matrix = np.array([[1, 0, 0, 0],
                                    [0, np.cos(theta), -np.sin(theta), 0],
                                    [0, np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 0, 1]])
        transform_matrix = rotation_matrix

    if theta_y != 0:

        theta = np.deg2rad(theta_y)
        rotation_matrix = np.asarray([[np.cos(theta), 0, np.sin(theta), 0],
                                      [0, 1, 0, 0],
                                      [-np.sin(theta), 0, np.cos(theta), 0],
                                      [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = rotation_matrix
        else:
            transform_matrix = np.dot(transform_matrix, rotation_matrix)

    if theta_z != 0:
        theta = np.deg2rad(theta_z)
        rotation_matrix = np.asarray([[np.cos(theta), -np.sin(theta), 0, 0],
                                      [np.sin(theta), np.cos(theta), 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = rotation_matrix
        else:
            transform_matrix = np.dot(transform_matrix, rotation_matrix)
    if tx != 0 or ty != 0 or tz != 0:
        shift_matrix = np.array([[1, 0, 0, tx],
                                 [0, 1, 0, ty],
                                 [0, 0, 1, tz],
                                 [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear_xy != 0:
        shear = np.deg2rad(shear_xy)
        shear_matrix = np.array([[1, -np.sin(shear), 0, 0],
                                 [0, np.cos(shear), 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if shear_xz != 0:
        shear = np.deg2rad(shear_xz)
        shear_matrix = np.array([[1, 0, -np.sin(shear), 0],
                                 [0, 1, 0, 0],
                                 [0, 0, np.cos(shear), 0],
                                 [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if shear_yz != 0:
        shear = np.deg2rad(shear_yz)
        shear_matrix = np.array([[1, 0, 0, 0],
                                 [0, 1, -np.sin(shear), 0],
                                 [0, 0, np.cos(shear), 0],
                                 [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1 or zz != 1:
        zoom_matrix = np.array([[zx, 0, 0, 0],
                                [0, zy, 0, 0],
                                [0, 0, zz, 0],
                                [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w, d = x.shape[row_axis], x.shape[col_axis], x.shape[z_axis]
        transform_matrix = transform_matrix_offset_center_fixed(
            transform_matrix, h, w, d)
        final_affine_matrix = transform_matrix[:3, :3]
        final_offset = transform_matrix[:3, 3]

        x = scipy.ndimage.interpolation.affine_transform(
            x,
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=cval)

    return x


def copy_template(cube, t, pos):
    """
    Returns a cube with the template t copied at position pos
    cube: 3D array of the cube
    t: 3D array of the template
    pos: [x y z] position in the cube
    """

    cube_size = cube.shape[0]
    margin = int(t.shape[0] / 2)  # if the position is on the side of the cube with this margin, the template won't be
    # copied entirely or it is outside the cube
    x_out1 = max(0, margin - pos[0])
    x_out2 = max(0, pos[0] - (cube_size - margin - 1))
    y_out1 = max(0, margin - pos[1])
    y_out2 = max(0, pos[1] - (cube_size - margin - 1))
    z_out1 = max(0, margin - pos[2])
    z_out2 = max(0, pos[2] - (cube_size - margin - 1))
    cube[max(0, pos[0] - margin):min(cube_size, pos[0] + margin + 1),
    max(0, pos[1] - margin):min(cube_size, pos[1] + margin + 1),
    max(0, pos[2] - margin):min(cube_size, pos[2] + margin + 1)] = t[x_out1:t.shape[0] - x_out2,
                                                                   y_out1:t.shape[1] - y_out2,
                                                                   z_out1:t.shape[2] - z_out2]
    return cube


def next_batch(num, data, labels, is_augment=False):
    """
    Returns a total of `num` random samples and labels.
    num: number of samples returned (batch size)
    data: volumes to sample from
    labels: ground truth labels to sample from
    augment: whether random 3D right-angle rotation (default=None)
    """
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    # randomly generate right-angle rotations
    if is_augment:
        xyz = np.array([[0, 0, 0], [90, 0, 0], [180, 0, 0], [270, 0, 0],
                        [0, 90, 0], [0, 90, 270], [0, 90, 180], [0, 90, 90],
                        [0, 180, 0], [90, 180, 0], [180, 180, 0], [270, 180, 0],
                        [0, 270, 0], [0, 270, 90], [0, 270, 180], [0, 270, 270],
                        [90, 0, 90], [180, 0, 90], [270, 0, 90], [0, 0, 90],
                        [90, 0, 270], [180, 0, 270], [270, 0, 270], [0, 0, 270]
                        ])
        nxyz = xyz.shape[0]
        random_angles = [xyz[np.random.randint(0, nxyz)] for i in idx]
        data_shuffle = [apply_affine_transform_fixed(np.squeeze(data[idx[i]]), theta_xyz=random_angles[i]) for i in
                        range(len(idx))]
        data_shuffle = np.expand_dims(data_shuffle, axis=-1)
    else:
        data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# Functions used for NLST only

def normalize(image, min_HU, max_HU):
    image = (image - min_HU) / (max_HU - min_HU)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def zero_center(image, mean_vox):
    image = image - mean_vox
    return image


def crop_center(vol, crop):
    x, y, z = vol.shape
    startx = x // 2 - (crop // 2)
    starty = y // 2 - (crop // 2)
    startz = z // 2 - (crop // 2)
    return vol[startx:startx + crop, starty:starty + crop, startz:startz + crop]


############################ Used for the NLST dataset only #######################################

def region_pool(name, X, Mask):
    """
    Returns global average pool in a region of the feature maps.
    X: feature maps
    Mask: binary mask
    """
    return tf.reduce_sum(tf.multiply(X, Mask), axis=[1, 2, 3]) / tf.reduce_sum(Mask, axis=[1, 2, 3])


def region_poolext(name, X, Mask):
    """
    Returns extended global average pool in a region of the feature maps.
    X: feature maps
    Mask: binary mask
    """
    Xsh = X.get_shape().as_list()
    bs = tf.shape(X)[0]
    mean_region = tf.reduce_sum(tf.multiply(X, Mask), axis=[1, 2, 3]) / tf.reduce_sum(Mask, axis=[1, 2, 3])
    size_region = tf.reduce_sum(Mask, axis=[1, 2, 3]) / int(Mask.shape[1]) ** 3

    # Reshape the feature maps and the mask to compute the variance.
    X = tf.transpose(tf.reshape(X, [bs, Xsh[1] * Xsh[2] * Xsh[3], Xsh[4]]), [1, 0, 2])
    Mask = tf.transpose(tf.reshape(Mask, [bs, Xsh[1] * Xsh[2] * Xsh[3], 1]), [1, 0, 2])

    var_region = tf.reduce_sum(tf.multiply((X - mean_region) ** 2, Mask), axis=[0]) / tf.reduce_sum(Mask, axis=[0])
    return tf.concat([mean_region, size_region, var_region], axis=1)


def region_poolsize(name, Mask):
    """
    Returns the size of the region only
    Mask: binary mask
    """
    size_region = tf.reduce_sum(Mask, axis=[1, 2, 3]) / int(Mask.shape[1]) ** 3
    return size_region


def maskReshape(Mask, ksize, stride):
    """
    Returns a reshaped version of the mask using avg pooling
    Mask: binary mask
    """
    return tf.nn.avg_pool3d(Mask, ksize=[1, ksize, ksize, ksize, 1], strides=[1, stride, stride, stride, 1],
                            padding='VALID')

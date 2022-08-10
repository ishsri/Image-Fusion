# from colors import rgb, hsv, hex, random
# from PIL import Image
# import cv2 as cv
# # pic = Image.open("image_1.1.png")
# # pic.show()

# weight_map = cv.imread("Fusion_R1/1.png")
# fusion = cv.imread(r"C:\Users\ishan\Desktop\Image Fusion\Fusion_R1\2.png")

# # hsl_weight = cv.cvtColor(weight_map, cv.COLOR_BGR2HLS)

# # hsl = cv.cvtColor(fusion, cv.COLOR_BGR2HLS)

# # cv.imshow("hsl", hsl_weight)
# # cv.waitKey(0)

import numpy as np
import tensorflow as tf
from PIL import Image
import scipy.misc


def rgb2hls(img):
    """ note: elements in img is a float number less than 1.0 and greater than 0.
    :param img: an numpy ndarray with shape NHWC
    :return:
    """
    assert len(img.shape) == 3
    hue = np.zeros_like(img[:, :, 0])
    luminance = np.zeros_like(img[:, :, 0])
    saturation = np.zeros_like(img[:, :, 0])
    for x in range(height):
        for y in range(width):
            r, g, b = img[x, y]
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            hue[x, y] = h
            luminance[x, y] = l
            saturation[x, y] = s
    return hue, luminance, saturation


def np_rgb2hls(img):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    maxc = np.max(img, -1)
    minc = np.min(img, -1)
    l = (minc + maxc) / 2.0
    if np.array_equal(minc, maxc):
        return np.zeros_like(l), l, np.zeros_like(l)
    smask = np.greater(l, 0.5).astype(np.float32)

    s = (1.0 - smask) * ((maxc - minc) / (maxc + minc)) + smask * ((maxc - minc) / (2.001 - maxc - minc))
    rc = (maxc - r) / (maxc - minc + 0.001)
    gc = (maxc - g) / (maxc - minc + 0.001)
    bc = (maxc - b) / (maxc - minc + 0.001)

    rmask = np.equal(r, maxc).astype(np.float32)
    gmask = np.equal(g, maxc).astype(np.float32)
    rgmask = np.logical_or(rmask, gmask).astype(np.float32)

    h = rmask * (bc - gc) + gmask * (2.0 + rc - bc) + (1.0 - rgmask) * (4.0 + gc - rc)
    h = np.remainder(h / 6.0, 1.0)
    return h, l, s


def tf_rgb2hls(img):
    """ note: elements in img all in [0,1]
    :param img: a tensor with shape NHWC
    :return:
    """
    assert img.get_shape()[-1] == 3
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    maxc = tf.reduce_max(img, -1)
    minc = tf.reduce_min(img, -1)

    l = (minc + maxc) / 2.0

    # if tf.reduce_all(tf.equal(minc, maxc)):
    #     return tf.zeros_like(l), l, tf.zeros_like(l)
    smask = tf.cast(tf.greater(l, 0.5), tf.float32)

    s = (1.0 - smask) * ((maxc - minc) / (maxc + minc)) + smask * ((maxc - minc) / (2.001 - maxc - minc))
    rc = (maxc - r) / (maxc - minc + 0.001)
    gc = (maxc - g) / (maxc - minc + 0.001)
    bc = (maxc - b) / (maxc - minc + 0.001)

    rmask = tf.equal(r, maxc)
    gmask = tf.equal(g, maxc)
    rgmask = tf.cast(tf.logical_or(rmask, gmask), tf.float32)
    rmask = tf.cast(rmask, tf.float32)
    gmask = tf.cast(gmask, tf.float32)

    h = rmask * (bc - gc) + gmask * (2.0 + rc - bc) + (1.0 - rgmask) * (4.0 + gc - rc)
    h = tf.mod(h / 6.0, 1.0)

    h = tf.expand_dims(h, -1)
    l = tf.expand_dims(l, -1)
    s = tf.expand_dims(s, -1)

    x = tf.concat([tf.zeros_like(l), l, tf.zeros_like(l)], -1)
    y = tf.concat([h, l, s], -1)

    return tf.where(condition=tf.reduce_all(tf.equal(minc, maxc)), x=x, y=y)


if __name__ == '__main__':
    """
    HLS: Hue, Luminance, Saturation
    H: position in the spectrum
    L: color lightness
    S: color saturation
    """
    avatar = Image.open("image_1.1.png")
    width, height = avatar.size
    print("width: {}, height: {}".format(width, height))
    img = np.array(avatar)
    img = img / 255.0
    print(img.shape)

    # # hue, luminance, saturation = rgb2hls(img)
    # hue, luminance, saturation = np_rgb2hls(img)

    img_tensor = tf.convert_to_tensor(img, tf.float32)
    hls = tf_rgb2hls(img_tensor)
    h, l, s = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        hue, luminance, saturation = sess.run([h, l, s])
        scipy.misc.imsave("hls_h_.jpg", hue)
        scipy.misc.imsave("hls_l_.jpg", luminance)
        scipy.misc.imsave("hls_s_.jpg", saturation)

r, g, b = cv2.split(image)
print(len(image))


def rgb_to_hsl (r, g, b):
    
    r/=255
    
    g/=255
    
    b/=255
    
    xmax = max(r,g,b)
    v = xmax
    xmin = min(r,g,b)

    c = xmax - xmin
    l = (xmax + xmin)*0.5
    
    if c == 0:
        h = 0
    elif v == r :
        h = 60 * (0 + ((g - b)/c))
    elif v == g :
        h = 60 * (2 + ((b - r)/c))
    elif v == b :
        h = 60 * (4 + ((r - g)/c))

    if l == 0 or l == 1:
        s = 0
    else :
        s = ((v - l))/(min(l , (1-l)))
    
    return (h, s, l)



def image_conv_1 (image) :
    r, g, b = cv2.split (image)
    hsl_image = np.zeros((len(image), len(image), 3))
    for i in range(0,len(image)):
        for j in range(0, len(image)):
            rr = r[i][j]
            bb = b[i][j]
            gg = g[i][j]

            h, s, l = rgb_to_hsl(rr, gg, bb)
            
            hsl_image[i][j] = [h, s, l]

    return hsl_image

hsl_image = image_conv_1(image)

print(hsl_image)

im = Image.fromarray(hsl_image.astype('uint8'), mode='HSV')

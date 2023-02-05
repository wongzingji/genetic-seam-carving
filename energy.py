import cv2
import numpy as np
from scipy import ndimage as ndi


# https://github.com/andrewdcampbell/seam-carving
def forward_energy(image):
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.
    Vectorized code adapted from
    https://github.com/axu2/improved-seam-carving.
    """
    h, w = image.shape[:2]
    g_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))

    U = np.roll(g_image, 1, axis=0)
    L = np.roll(g_image, 1, axis=1)
    R = np.roll(g_image, -1, axis=1)

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, h):
        mU = m[i - 1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)

    # vis = visualize(energy)
    # cv2.imwrite("forward_energy_demo.jpg", vis)

    return energy / 255.0

# https://github.com/andrewdcampbell/seam-carving
def backward_energy(image):
    """
    Simple gradient magnitude energy map.
    """
    xgrad = ndi.convolve1d(image, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(image, np.array([1, 0, -1]), axis=0, mode='wrap')

    grad_mag = np.sqrt(np.sum(xgrad ** 2, axis=2) + np.sum(ygrad ** 2, axis=2))

    # vis = visualize(grad_mag)
    # cv2.imwrite("backward_energy_demo.jpg", vis)

    return grad_mag / 255.0

def saliency_spectral_residual(img):
    """
    saliency maps
    """
    _, saliency_map = cv2.saliency.StaticSaliencySpectralResidual_create().computeSaliency(img)
    return (saliency_map * 255).astype('uint8')

def saliency_fine_grained(img):
    """
    saliency maps
    """
    _, saliency_map = cv2.saliency.StaticSaliencyFineGrained_create().computeSaliency(img)
    return (saliency_map * 255).astype('uint8')
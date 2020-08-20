import numpy as np

def _assert_image_shapes_equal(org_img: np.ndarray, pred_img: np.ndarray, metric: str):
    msg = (f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
           f"{str(org_img.shape)}, y_pred shape = {str(pred_img.shape)}")

    assert org_img.shape == pred_img.shape, msg


def psnr(org_img: np.ndarray, pred_img: np.ndarray, data_range=4096):
    """
    Peek Signal to Noise Ratio, a measure similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When using 12-bit imagery MaxP is 4096, for 8-bit imagery 256
    """
    _assert_image_shapes_equal(org_img, pred_img, "PSNR")

    r = []
    for i in range(org_img.shape[0]):
        val = 20 * np.log10(data_range) - 10. * np.log10(np.mean(np.square(org_img[i, :, :] - pred_img[i, :, :])))
        r.append(val)

    return np.mean(r)

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def uiq(org_img: np.ndarray, pred_img: np.ndarray):
    """
    Universal Image Quality index
    """
    _assert_image_shapes_equal(org_img, pred_img, "UIQ")
    q_all = []
    for (x, y, window_org), (x, y, window_pred) in zip(sliding_window(org_img, stepSize=1, windowSize=(8, 8)),
                                                       sliding_window(pred_img, stepSize=1, windowSize=(8, 8))):
        # if the window does not meet our desired window size, ignore it
        if window_org.shape[0] != 8 or window_org.shape[1] != 8:
            continue
        org_img_mean = np.mean(org_img)
        pred_img_mean = np.mean(pred_img)
        org_img_variance = np.var(org_img)
        pred_img_variance = np.var(pred_img)
        org_pred_img_variance = np.mean((window_org - org_img_mean) * (window_pred - pred_img_mean))

        numerator = 4 * org_pred_img_variance * org_img_mean * pred_img_mean
        denominator = (org_img_variance + pred_img_variance) * (org_img_mean**2 + pred_img_mean**2)

        if denominator != 0.0:
            q = numerator / denominator
            q_all.append(q)

    return np.mean(q_all)


def sam(org_img: np.ndarray, pred_img: np.ndarray):
    """
    calculates spectral angle mapper
    """
    _assert_image_shapes_equal(org_img, pred_img, "SAM")
    org_img = org_img.reshape((org_img.shape[0] * org_img.shape[1], org_img.shape[2]))
    pred_img = pred_img.reshape((pred_img.shape[0] * pred_img.shape[1], pred_img.shape[2]))

    N = org_img.shape[1]
    sam_angles = np.zeros(N)
    for i in range(org_img.shape[1]):
        val = np.clip(np.dot(org_img[:, i], pred_img[:, i]) / (np.linalg.norm(org_img[:, i]) * np.linalg.norm(pred_img[:, i])), -1, 1)
        sam_angles[i] = np.arccos(val)

    return np.mean(sam_angles)


def sre(org_img: np.ndarray, pred_img: np.ndarray):
    """
    signal to reconstruction error ratio
    """
    _assert_image_shapes_equal(org_img, pred_img, "SRE")

    sre_final = []
    for i in range(org_img.shape[2]):
        numerator = (np.mean(org_img[:, :, i]))**2
        denominator = ((np.linalg.norm(org_img[:, :, i] - pred_img[:, :, i]))) /\
                      (org_img.shape[0] * org_img.shape[1])
        sre_final.append(10 * np.log10(numerator/denominator))

    return np.mean(sre_final)
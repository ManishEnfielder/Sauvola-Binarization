import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import integral_image
from tqdm import tqdm
import warnings

def get_window_sum(cum_sum, w):
    window_sums = np.zeros((cum_sum.shape[0] - w[0] + 1, cum_sum.shape[1] - w[1] + 1))
    for i in tqdm(range(len(window_sums))):
        for j in range(window_sums.shape[1]):
            window_sums[i, j] = (
                cum_sum[i + w[0] - 1, j + w[1] - 1]
                - cum_sum[i + w[0] - 1, j]
                - cum_sum[i, j + w[1] - 1]
                + cum_sum[i, j]
            )
    return window_sums


def get_mean_std(image, w):

    pad_width = w // 2
    padded = np.pad(image, pad_width, mode="reflect")
    
    cum_sum = integral_image(padded)
    padded = np.power(padded, 2)
    cum_sum_squared = integral_image(padded)

    total_window_size = w ** 2
    w = (w, w)
    mean = get_window_sum(cum_sum, w)
    mean /= total_window_size
    squared_mean = get_window_sum(cum_sum_squared, w)
    squared_mean /= total_window_size

  
    std = np.sqrt(np.clip(squared_mean - mean ** 2, 0, None))
    return mean, std


def sauvola(image, window_size=15, k=0.2, r=None):

    if r is None:
        imin, imax = image.min(), image.max()
        r = 0.5 * (imax - imin)
    mean, std = get_mean_std(image, window_size)
    return mean * (1 + k * ((std / r) - 1))


if __name__ == "__main__":
    image = plt.imread("Path to read")
    image = rgb2gray(image)
    mask = image > sauvola(image)
    image[mask] = 0
    image[~mask] = 1
    warnings.filterwarnings('ignore')
    plt.imshow(image, cmap="Greys")
    plt.imsave("Path to save", image)
    plt.waitforbuttonpress()

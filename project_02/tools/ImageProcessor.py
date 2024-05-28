import cv2 as cv
import numpy as np
import torchvision


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple[0], original_tuple[1], path)
        return tuple_with_path

def gradient_graph(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Compute the x and y gradients using the Sobel operator
    grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=1)
    grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=1)

    # Compute the gradient magnitude
    grad_magnitude = cv.magnitude(grad_x, grad_y)

    # Normalize the gradient image to 0-255
    grad_magnitude = cv.normalize(grad_magnitude, None, 0, 255, cv.NORM_MINMAX)

    # Convert to 8-bit image
    gradient_image = np.uint8(grad_magnitude)
    gradient_image = adjust_contrast(gradient_image, 3.0, 0)
    return gradient_image
    

# read image
def read_image(image_path):
    image = cv.imread(image_path)
    # image = _shrink(image, 100)
    # image = adjust_contrast(image, 2.0, -20)
    return image

def resize(image, width, height):
    return cv.resize(image, (width, height))

def _resize(img, d):
    return resize(img, d, d)

def shrink(img, ratio):
    return resize(img, int(img.shape[1] * ratio), int(img.shape[0] * ratio))

def _shrink(img, height):
    return shrink(img, height/ img.shape[0])

# 调整对比度
def adjust_contrast(image, alpha, beta):
    """
    Adjust the contrast and brightness of an image.

    Parameters:
        image (numpy.ndarray): The input image.
        alpha (float): Contrast control (1.0-3.0).
        beta (int): Brightness control (0-100).

    Returns:
        numpy.ndarray: The adjusted image.
    """
    # Convert the image to float32 to prevent clipping values
    adjusted = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def compute_gradient_image(image):
    """
    Compute the gradient (edge) image using Canny edge detection.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The gradient image.
    """
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv.GaussianBlur(gray, (1, 1), 0)

    # Use Canny edge detection
    edges = cv.Canny(blurred, 30, 65)  # Adjust these thresholds as needed

    return edges


# test
# image = read_image("../statics/FundusDomainTest/1/gdrishtiGS_020.png")
# grad = gradient_graph(image)
# edge = compute_gradient_image(image)
# #show
# cv.imshow("image", grad)
# cv.waitKey(0)
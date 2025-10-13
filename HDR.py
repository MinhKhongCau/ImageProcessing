import cv2
import numpy as np

def readImage(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'image not load')
    return img

def comnineExposure(images):
    high_quality = np.zeros_like(images[0], dtype=np.float32)

    weight = 1.0 / len(images)

    for image in images:
        high_quality += image.astype(np.float32) * weight

    return np.clip(high_quality, 0, 255).astype(np.uint8)

    

def showImage(window_name, img):
    print('shape', img.shape)
    print('dtype', img.dtype)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__': 

    images = []

    images.append(cv2.imread('palace-8.png'))
    images.append(cv2.imread('palace-2.png'))
    images.append(cv2.imread('palace+2.png'))
    images.append(cv2.imread('palace+4.png'))

    if len(images) > 0:
        showImage('Demo image',images[0])
        showImage('Demo image',images[1])
        showImage('Demo image',images[2])
        showImage('Demo image',images[3])

        imageclearly = comnineExposure(images=images)
        showImage("Noise Reduction", imageclearly)
    else:
        print("No images captured!")

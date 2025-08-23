import cv2
import numpy as np

def readImage(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'image not load')
    return img

def noiseRedution(images):
    w, h, channel = images[0].shape
    sumImage = np.zeros_like(images[0], dtype=np.float32)
    print(sumImage.shape)
    for i in images:
        sumImage += i
    redution = sumImage / len(images)

    return redution.astype(np.uint8)

def showImage(window_name, img):
    print('shape', img.shape)
    print('dtype', img.dtype)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__': 
    video = cv2.VideoCapture(0)

    images = []

    for i in range(0,60,1):
        ret, image = video.read()

        if not ret:
            continue

        images.append(image)
    
    video.release()

    if len(images) > 0:
        showImage('Noise image',images[0])
        imageclearly = noiseRedution(images=images)
        showImage("Noise Reduction", imageclearly)
    else:
        print("No images captured!")
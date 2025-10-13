import cv2
import matplotlib.pyplot as plt

def readImage(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'image not load')
    return img
   

def histogram(image):
    # charge to gray scale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # painting
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(gray.ravel(),bins=256,range=(0,256))
    plt.title("Histogram")
    plt.xlabel("Gray level")
    plt.ylabel("Pixel")

    plt.show()

def showImage(window_name, img):
    print('shape', img.shape)
    print('dtype', img.dtype)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    image = readImage('bay.jpg')
    histogram(image)



import cv2

SIZE_500X500 = (500, 500)
SIZE_100X100 = (100, 100)
SIZE_50X50 = (50,50)
SIZE_25X25 = (25, 25)


def readImage(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'image not load')
    return img

def resizeImage(img, size):
    return cv2.resize(img, size)

def showImage(window_name, img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img = readImage("Labrador_Retriever_portrait.jpg")
    resized_img_25 = resizeImage(img, SIZE_25X25)
    resized_img_50 = resizeImage(img, SIZE_50X50)
    resized_img_100 = resizeImage(img, SIZE_100X100)
    resized_img_500 = resizeImage(img, SIZE_500X500)
    showImage("Resized 25x25", resized_img_25)
    showImage("Resized 50x50", resized_img_50)
    showImage("Resized 100x100", resized_img_100)
    showImage("Resized 500x500", resized_img_500)

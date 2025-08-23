import cv2

def readImage(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'image not load')
    return img

def subtraction(backgroundImage, itemImage):
    substracted = cv2.subtract(backgroundImage, itemImage)
    return substracted


def showImage(window_name, img):
    print('shape', img.shape)
    print('dtype', img.dtype)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mergeCircleAndStar():
    background = cv2.imread('star.png')
    itemImage = cv2.imread('circle.png')

    showImage('background',background)
    showImage('item',itemImage)

    showImage('substracted', subtraction(backgroundImage=background, itemImage=itemImage))

def detectItem():
    # Mở webcam (0 là camera mặc định)
    video = cv2.VideoCapture(0)

    # Đọc frame đầu tiên làm background
    ret, background = video.read()
    if not ret:
        print("Không lấy được background từ camera")
        return

    # Đợi 2s cho camera ổn định
    cv2.waitKey(2000)

    while True:
        ret, item = video.read()
        if not ret:
            break

        rel = subtraction(background, item)

        cv2.imshow('video', rel)

        key = cv2.waitKey(25) & 0xFF 
        if key == ord('q'): # thoát bằng phím 'q'
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__': 
    mergeCircleAndStar()

    # detectItem()

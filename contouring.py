import cv2
import numpy as np

def readImage(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'image not load')
    return img

def quantize_bits(img, bits):
    levels = 2 ** bits 
    quantized = np.floor(img / (256 / levels)) * (255 / (levels - 1))
    return quantized.astype(np.uint8)

def bit_slicing(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    for bit in range(8, 0, -1):
        q_img = quantize_bits(img_gray, bit)

        cv2.imshow(f'Image ${bit+1} bit', q_img)
       
    showImage('origin image ', img_gray)
        
def showImage(window_name, img):
    print('shape', img.shape)
    print('dtype', img.dtype)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img = readImage("Screenshot from 2025-08-18 22-41-46.png")

    showImage('Origin image', img)
    bit_slicing(img)

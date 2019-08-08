import faceSwap
import cv2

image1 = "images/yony.jpg"
image2 = "images/anchorman.jpg"

img1 = cv2.imread(image1)
img2 = cv2.imread(image2)

faceSwap = faceSwap.FaceSwap()
faceSwap.faceSwap(img1, img2, mode="choose_largest_face",
                  showImages=True)

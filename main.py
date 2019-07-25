import faceSwap
import os

image1 = input("Enter the first image name: ")
image2 = input("Enter the second image name: ")

path = "images/original_images/"

image1path = os.path.join(path, image1)
image2path = os.path.join(path, image2)

img1 = cv2.imread(image1path)
img2 = cv2.imread(image2path)

faceSwap = faceSwap.FaceSwap()
faceSwap.faceSwap(img1, img2,
                  showImages=True, saveSwappedImage=False)

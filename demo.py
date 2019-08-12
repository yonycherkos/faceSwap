import faceSwap
import cv2
import argparse

parser = argparse.ArgumentParser(description='Enter image path of input and meme images.')

parser.add_argument('image1', help="Eneter image1 file path", type=str, default="images/yony.jpg")
parser.add_argument('image2', help="Enter image2 file path", type=str, default="images/anchorman.jpg")

args = parser.parse_args()

img1 = cv2.imread(args.image1)
img2 = cv2.imread(args.image2)

faceSwap = faceSwap.FaceSwap()
faceSwap.faceSwap(img1, img2, mode="choose_largest_face",
                  showImages=True)

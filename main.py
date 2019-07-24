import faceSwap

# image1 = input("Enter the first image file path: ")
# image2 = input("Enter the second image file path: ")

image1 = "images/original_images/anchorman.jpg"
image2 = "images/original_images/yony.jpg"

faceSwap = faceSwap.FaceSwap(image1, image2)
faceSwap.faceSwap(showImages=True, saveSwappedImage=False)

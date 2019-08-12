# FaceSwap


This is face swapping project with opencv python. It takes two images as input then replace the face part of the first image into the face face part of the second one. if there are multiple faces in one of the images, it use the largest face.

## Prerequisites

- For this project you need to install python3 and necessary libraries using the following command:

  - `pip3 install -r requirements.txt`

* Download the facial landmark detector dlib model from [here](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat) and put it in the main project folder. Or alternatively, use the following command from main project folder to get the file:
  - `wget https://github.com/AKSHAYUBHAT/TensorFace/raw/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat`


## Getting Started

For getting start with this project
* select two image from the given one or your own images.
* then run the demo.py python script and Enter image1 and image2 absolute or relative path as argument.
  and choose the modes either choose_largest_face or apply_on_all mode.
* ex: python demo.py images/yony.jpg images/anchorman.jpg choose_largest_face
* the output show the two original images and the swapped image based on your choose.

## how to use the grpc client and server
To test the grpc client and server interaction follow these steps:
* first run the start_server.py.
* then run client.py from /Service/client.py
* and see the swapped image output on /Service/images/output.img
* open /Service/client.py [optional]
* then change the the first image and meme image path to your own image path. [optional]
* you can also change the output image path. [optional]

## Authors

* **yonathan cherkos, Israel Abebe, and Yared Taddese**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

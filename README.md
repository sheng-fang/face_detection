# Face Detection
Create a face detection model from scratch with Tensorflow 2

Project starts on 2020-04-22. The objective of this project is to create a face detection model from scratch and build a 
local Tensorflow service.  

1. Model: RetinaFace
2. Database: Wider FACE (http://shuoyang1213.me/WIDERFACE/)
3. Hardware: GTX 1080, 8G

Because face detection is a branch of generic object detection, I build Yolo and SSD object detection model and train 
them on  WIDER FACE dataset. Then the last model RetinaFace is realized by jointly learn landmark of face and 
localizations of faces at the same time. 



Reference:

https://github.com/ChunML/ssd-tf2
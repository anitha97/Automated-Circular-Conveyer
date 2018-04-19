import socket 
import RPi.GPIO as GPIO 
import time
from time import sleep 

import cv2
#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as 
#it is needed by OpenCV face recognizers
import numpy as np
import io
import picamera


GPIO.setmode(GPIO.BOARD) 
GPIO.setwarnings(False) 
#GPIO.cleanup()
#GPIO.setup(19,GPIO.OUT)  #button
GPIO.setup(03,GPIO.OUT)  #e1 
GPIO.setup(05,GPIO.OUT)  #e2 
GPIO.setup(07,GPIO.OUT)   #driver1 
GPIO.setup(11,GPIO.OUT) 
GPIO.setup(8,GPIO.OUT)   #driver2 
GPIO.setup(10,GPIO.OUT) 
GPIO.setup(12,GPIO.IN)  #left sensor 
GPIO.setup(13,GPIO.IN)  #right_sensor
GPIO.setup(15,GPIO.IN)  #left object
GPIO.setup(16,GPIO.IN)  #right_object
#GPIO.setup(18,GPIO.OUT)  #buzzer
GPIO.setup(19,GPIO.IN)   #button
pwm1= GPIO.PWM(03, 100) 
pwm2= GPIO.PWM(05, 100) 
#GPIO.output(03,GPIO.HIGH) 
#GPIO.output(05,GPIO.HIGH)
pwm1.start(0) 
pwm2.start(0)
host=""
port= 6000

face_count_auth=0

#GPIO.output(18,GPIO.LOW) #buzzer off initially
subjects = ["", "aravind","manisha","thusar sir","siva ranjini madam","gaayathri madam","raju","ramesh"]


def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('haarcas')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        print("face not detected")
        return None, None
    else:
        print("face detected")
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


# I am using OpenCV's **LBP face detector**. On _line 4_, I convert the image to grayscale because most operations in OpenCV are performed in gray scale, then on _line 8_ I load LBP face detector using `cv2.CascadeClassifier` class. After that on _line 12_ I use `cv2.CascadeClassifier` class' `detectMultiScale` method to detect all the faces in the image. on _line 20_, from detected faces I only pick the first face because in one image there will be only one face (under the assumption that there will be only one prominent face). As faces returned by `detectMultiScale` method are actually rectangles (x, y, width, height) and not actual faces images so we have to extract face image area from the main image. So on _line 23_ I extract face area from gray image and return both the face image area and face rectangle.
# 
# Now you have got a face detector and you know the 4 steps to prepare the data, so are you ready to code the prepare data step? Yes? So let's do it. 

# In[4]:

#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
    
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
        
        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;
            
        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
        
        #build path of directory containin images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
            
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)
            
            #display an image window to show the image 
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            
            #detect face
            face, rect = detect_face(image)
            
            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if face is not None:
                print("yes got the face!!")
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels


# I have defined a function that takes the path, where training subjects' folders are stored, as parameter. This function follows the same 4 prepare data substeps mentioned above. 
# 
# **(step-1)** On _line 8_ I am using `os.listdir` method to read names of all folders stored on path passed to function as parameter. On _line 10-13_ I am defining labels and faces vectors. 
# 
# **(step-2)** After that I traverse through all subjects' folder names and from each subject's folder name on _line 27_ I am extracting the label information. As folder names follow the `sLabel` naming convention so removing the  letter `s` from folder name will give us the label assigned to that subject. 
# 
# **(step-3)** On _line 34_, I read all the images names of of the current subject being traversed and on _line 39-66_ I traverse those images one by one. On _line 53-54_ I am using OpenCV's `imshow(window_title, image)` along with OpenCV's `waitKey(interval)` method to display the current image being traveresed. The `waitKey(interval)` method pauses the code flow for the given interval (milliseconds), I am using it with 100ms interval so that we can view the image window for 100ms. On _line 57_, I detect face from the current image being traversed. 
# 
# **(step-4)** On _line 62-66_, I add the detected face and label to their respective vectors.

# But a function can't do anything unless we call it on some data that it has to prepare, right? Don't worry, I have got data of two beautiful and famous celebrities. I am sure you will recognize them!
# 
# ![training-data](visualization/tom-shahrukh.png)
# 
# Let's call this function on images of these beautiful celebrities to prepare data for training of our Face Recognizer. Below is a simple code to do that.

# In[5]:

#let's first prepare our training data
#data will be in two lists of same size
#one list will contain all the faces
#and other list will contain respective labels for each face
def train_data():
    print("Preparing data ")
    faces, labels = prepare_training_data("training-data")
    print("Data prepared")
    #print total faces and labels
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    #create our LBPH face recognizer 
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    #or use EigenFaceRecognizer by replacing above line with 
    #face_recognizer = cv2.face.EigenFaceRecognizer_create()

    #or use FisherFaceRecognizer by replacing above line with 
    #face_recognizer = cv2.face.FisherFaceRecognizer_create()


    # Now that we have initialized our face recognizer and we also have prepared our training data, it's time to train the face recognizer. We will do that by calling the `train(faces-vector, labels-vector)` method of face recognizer. 

    #train our face recognizer of our training faces
    face_recognizer.train(faces, np.array(labels)) 
    return face_recognizer


# This was probably the boring part, right? Don't worry, the fun stuff is coming up next. It's time to train our own face recognizer so that once trained it can recognize new faces of the persons it was trained on. Read? Ok then let's train our face recognizer. 

# ### Train Face Recognizer

# As we know, OpenCV comes equipped with three face recognizers.
# 
# 1. EigenFace Recognizer: This can be created with `cv2.face.createEigenFaceRecognizer()`
# 2. FisherFace Recognizer: This can be created with `cv2.face.createFisherFaceRecognizer()`
# 3. Local Binary Patterns Histogram (LBPH): This can be created with `cv2.face.LBPHFisherFaceRecognizer()`
# 
# I am going to use LBPH face recognizer but you can use any face recognizer of your choice. No matter which of the OpenCV's face recognizer you use the code will remain the same. You just have to change one line, the face recognizer initialization line given below. 

# In[6]:


#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject

def identify_person(label):

    if(label==-1):
        print("Unauthorised person")
        os.system('espeak "Sorry! Unauthorised person "')
        return 0

    elif(label==1):
        print("aravind")
        os.system('espeak "Hello aravind "')

    elif (label==2):
        print("manisha")
        os.system('espeak "Hello manisha "')

    elif (label==3):
        print("thusar sir")
        os.system('espeak "Hello thusar sir "')

    elif (label==4):
        print("shiva ranjini madam")
        os.system('espeak "Hello siva ranjini madam "') 

    elif (label==5):
        print("gaayathri madam")
        os.system('espeak "Hello gaayathri madam "')

    elif (label==6):
        print("Raju")
        os.system('espeak "Hello raju"')

    elif (label==7):
        print("Ramesh")
        os.system('espeak "Hello ramesh"')

       

    return 1





def predict(face_recognizer,test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)
    if(face is None):
        #print("face not detected2")
        return 0;
    print("Got your picture")
    os.system('espeak "face captured.. "')
    #predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
    #get name of respective label returned by face recognizer
    label_text = subjects[label]
    vrfy=identify_person(label)
    while(vrfy==0):
        face_count_auth+=1
        if(face_count_auth>=3):
          face_count_auth=0
          os.system('espeak "Exceeded the limit! Thank you ! Have a nice day! Bye Bye "')
          return -1
        os.system('espeak "it\'s ok! we will verify you again! Please stand in position "')
        sleep(1)
        camera.capture('pic1.jpg')
        image1=cv2.imread('pic1.jpg')
        
        img = imag1e.copy()
        #detect face from the image
        face, rect = detect_face(img)
        if(face is None):
          #print("face not detected2")
          return 0;
        print("Got your picture")
        os.system('espeak "face captured.. "')
        label, confidence = face_recognizer.predict(face)
        #get name of respective label returned by face recognizer
        label_text = subjects[label]
        vrfy=identify_person(label)

    os.system('espeak "Here\'s your circular"')
    # CAN INCLUDE FUNCTION HERE TO PROCESS CIRCULAR "
    os.system("./pocketsphinx_continuous -lm 3518.lm -dict 3518.dic")
    return -1
    #draw a rectangle around face detected
    #draw_rectangle(img, rect)
    #draw name of predicted person
    #draw_text(img, label_text, rect[0], rect[1]-5)
    
    #return img

# Now that we have the prediction function well defined, next step is to actually call this function on our test images and display those test images to see if our face recognizer correctly recognized them. So let's do it. This is what we have been waiting for. 

# In[10]:

def start_take_pic(face_rec):
    print("Taking image...")
    #stream = io.BytesIO()
    #Get the picture (low resolution, so it should be quite fast)
    #Here you can also specify other parameters (e.g.:rotate the image)
    camera= picamera.PiCamera() 
    camera.resolution = (720,1024)
    camera.start_preview()
    sleep(2)
    camera.capture('pic1.jpg')
    image1=cv2.imread('pic1.jpg')
    print("face captured")
    predicted_img1 = predict(face_rec,image1)
    while(predicted_img1 == 0):
           print("face not identified,trying again")
           os.system('espeak "unable to detect your face , please stand in correct position"')
           sleep(1)
           camera.capture('pic1.jpg')
           image1=cv2.imread('pic1.jpg')
           predicted_img1 = predict(face_rec,image1)
          
    print("Prediction complete")
    camera.stop_preview()
    camera.close()
    if(predicted_img1 == -1):            #write code for continuing its opearion3
        return;





def helloworld():
    os.system('espeak -v+f3 -f speak11.wav')
    sleep(0.5)
    os.system('espeak -v+f3 -f speak12.wav')
    sleep(0.3)
    #os.system('espeak -v+f3 -f speak13.wav')
    #os.system('espeak -v+f3 -f speak14.wav')
    #sleep(0.3)

def f(): 
    GPIO.output(07,GPIO.HIGH) 
    GPIO.output(11,GPIO.LOW) 
    GPIO.output(10,GPIO.HIGH) 
    GPIO.output(8,GPIO.LOW)
    pwm1.ChangeDutyCycle(25) 
    pwm2.ChangeDutyCycle(25)
    

    
def left(): 
    GPIO.output(07,GPIO.HIGH) 
    GPIO.output(11,GPIO.LOW) 
    GPIO.output(10,GPIO.HIGH) 
    GPIO.output(8,GPIO.LOW)
    pwm1.ChangeDutyCycle(80) 
    pwm2.ChangeDutyCycle(0) 
    
   # sleep(2)  
def right(): 
    GPIO.output(07,GPIO.HIGH) 
    GPIO.output(11,GPIO.LOW) 
    GPIO.output(10,GPIO.HIGH) 
    GPIO.output(8,GPIO.LOW)
    pwm1.ChangeDutyCycle(0) 
    pwm2.ChangeDutyCycle(80) 
   
def frontlean():
    GPIO.output(07,GPIO.HIGH) 
    GPIO.output(11,GPIO.LOW) 
    GPIO.output(10,GPIO.HIGH) 
    GPIO.output(8,GPIO.LOW)
    pwm1.ChangeDutyCycle(25) 
    pwm2.ChangeDutyCycle(25)
    sleep(0.3)
    
def buzzertest():
  while True:
   buz=GPIO.input(19)
   if buz :
     print "busser vale ",buz
     return 1
    
def start(face_rec): 
  while True:
     #GPIO.output(18,GPIO.LOW)
     #lsensor_input=  GPIO.input(12) 
     #rsensor_input=  GPIO.input(13)
     if(GPIO.input(15) == 0 or GPIO.input(16)==0):
       pwm1.ChangeDutyCycle(0) 
       pwm2.ChangeDutyCycle(0)
       #s_time=time.time()
       while(GPIO.input(15) == 0 or GPIO.input(16)==0):
               os.system('espeak "side please"')
               sleep(0.6)
     if(GPIO.input(12) == 0 and GPIO.input(13)==0): 
              f()
         
     elif(GPIO.input(12)== 1 and GPIO.input(13)==1): 
              pwm1.ChangeDutyCycle(0) 
              pwm2.ChangeDutyCycle(0)
              for i in range(2):
                os.system('espeak -v+f3 -f speak3.wav')
                sleep(0.6)
              os.system("./pocketsphinx_continuous -lm 3518.lm -dict 3518.dic")
              #sleep(2) # time for the person to stand in positon to camera
              start_take_pic(face_rec)
              sleep(5)
              frontlean()

     elif(GPIO.input(12) == 1) : 
              left() 
     elif(GPIO.input(13) ==1) : 
              right() 
  print "stopped!!!!!!!!!" 
  pwm1.ChangeDutyCycle(0) 
  pwm2.ChangeDutyCycle(0) 
 
def setup():
  s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
  print "socket created"
  try:
    s.bind((host,port))
  except socket.error as msg:
    print msg
  print "socket bind complete"
  return s

def setupconn():
  s.listen(1)
  conn,address=s.accept()
  print "connected to : ",address[0]
  return conn
def dataTransfer(conn):
  data=conn.recv(1024)
  print "data treceived from android ",data
  if(data == "1"):
     helloworld()
     face_rec=train_data()
     print("start action")
     start(face_rec)
 #print "received 1"
  elif data=="0" :
     print "Stop"
     start(0)

s=setup()
while True:
  try:
   conn=setupconn()
   dataTransfer(conn)
  except:
   break
pwm1.ChangeDutyCycle(0) 
pwm2.ChangeDutyCycle(0) 
GPIO.cleanup()


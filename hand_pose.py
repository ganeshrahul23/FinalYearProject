import cv2
import numpy as np
import util as ut
import svm_train as st
import time
from pygame import mixer
import threading

model=st.trainSVM(5,20,'TrainData')
move_text={'0':'ZERO','1':'ONE','2':'TWO','3':'THREE','4':'FOUR','5':'FIVE','6':'SIX','7':'SEVEN','8':'EIGHT','9':'NINE'}
aud_fil=['0.mp3','1.mp3','2.mp3','3.mp3','4.mp3','5.mp3','6.mp3','7.mp3','8.mp3','9.mp3']
cam = 0
cap=cv2.VideoCapture(cam)
font = cv2.FONT_HERSHEY_SIMPLEX
mixer.init()

while(cap.isOpened()):
        global lock
        lock=threading.Lock()
        move=''
        _,img=cap.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,th1 = cv2.threshold(gray.copy(),150,255,cv2.THRESH_TOZERO)
        cv2.imshow('thresh',th1)
        _,contours,_ = cv2.findContours(th1.copy(),cv2.RETR_EXTERNAL, 2)
        cnt=ut.getMaxContour(contours,4000)
        if cnt!=None:
                _,res=ut.getGestureImg(cnt,img,th1,model)
                cv2.imshow('PredictedGesture',cv2.imread('TrainData/'+res+'_1.jpg'))
                move=move_text[res]
				mixer.music.load('Aud/'+aud_fil[int(res)])
				mixer.music.play()
				mixer.music.play()            
        cv2.putText(img,"Gesture: "+move,(50,50), font,1,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('Frame',img)
        k = 0xFF & cv2.waitKey(10)
        if k == 27:
                break       
cap.release()        
cv2.destroyAllWindows()

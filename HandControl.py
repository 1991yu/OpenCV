import cv2
import itertools
import mediapipe as mp
import numpy as np
import random
from PIL import Image
from playsound import playsound
import pygame
import os
import tkinter as tk
from tkinter import filedialog
import shutil




# 定义了一个手部识别的工具,手势识别类
class HandDetector():
    #初始化方法
    def __init__(self):
        #手势识别器，读取的是mediapipe的手部识别解决方案,两个属性
        self.hand_detector  = mp.solutions.hands.Hands()
        self.drawer = mp.solutions.drawing_utils

    #处理mediapipe的手部数据，并将其显示出来
    def process(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #将图片从BGR格式变成RGB格式
        self.hands_data = self.hand_detector.process(img_rgb)#将获取到的数据保存到成员变量中
        #print(result.multi_hand_landmarks)  

        if self.hands_data.multi_hand_landmarks:  # 如果有获取到手势数据的话
            #循环遍历这部分的数据
            for handlms in self.hands_data.multi_hand_landmarks:
                #画出手指的各个节点和连线
                self.drawer.draw_landmarks(img, handlms, mp.solutions.hands.HAND_CONNECTIONS)
        
    def find_position(self, img):  #获取识别是左手还是右手
        h, w, c = img.shape #或许视频的长度和高度,c没用上
        #print(h,w,c)
        #创建两个字典,分别存储左右手的数据
        position = {'Left':{},'Right':{}}
        if self.hands_data.multi_hand_landmarks: #有手部数据的话
            for i, point in enumerate(self.hands_data.multi_handedness):    #遍历每一个数据

                score = point.classification[0].score  #百分之多少是左手或右手

                if score >=0.8:#如果准确率大于80%

                    label = point.classification[0].label 
                    #获取到左右手的数据,label为左右手
                    hand_lms = self.hands_data.multi_hand_landmarks[i].landmark 
                    #返回的是一个列表，包含每个手部关节点的数据（一共21个）

                    for id , lm in enumerate(hand_lms): #遍历这些数据
                        x, y = int(lm.x * w), int(lm.y * h) 
                        #获取每个关节在视频中的坐标
                        position[label][id] = (x, y)  
                        #创建了一个元组就是说，label（是左手还是右手），id（第几号节点）,(x,y)他的坐标是多少
                        #print(position)
        return position #返回字典



# 图片叠加功能模块
def picplus(img1,img2,coordinate):
    img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)) #转换为PIL格式
    img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    img1_pil.paste(img2_pil, coordinate) #img2贴在img1指定位置，位置是(左,上)
    return cv2.cvtColor(np.asarray(img1_pil), cv2.COLOR_RGB2BGR)


#图片显示功能模块
def image_plus(img_back,img,center):

    # img=cv2.imread('zhadan.jpg')
    # img_back=cv2.imread('sky.jpg')
    #日常缩放

    rows,cols,channels = img_back.shape
    img_back=cv2.resize(img_back,None,fx=1,fy=1)
    #cv2.imshow('img_back',img_back)

    rows,cols,channels = img.shape
    img=cv2.resize(img,None,fx=1,fy=1)

    #cv2.imshow('img',img)
    rows,cols,channels = img.shape#rows，cols最后一定要是前景图片的，后面遍历图片需要用到

    #转换hsv
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #获取mask
    lower_blue=np.array([78,43,46])
    upper_blue=np.array([110,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #cv2.imshow('Mask', mask)

    #腐蚀膨胀
    erode=cv2.erode(mask,None,iterations=1)
    #cv2.imshow('erode',erode)

    dilate=cv2.dilate(erode,None,iterations=1)
    #cv2.imshow('dilate',dilate)

    #遍历替换
    #center=[50,50]#在新背景图片中的位置
    for i, j in itertools.product(range(rows), range(cols)):
        if dilate[i,j]==0:#0代表黑色的点
            img_back[center[1]+i,center[0]+j]=img[i,j]#此处替换颜色，为BGR通道

    #cv2.imshow('res',img_back) 

    return(img_back)



#两点间距离计算模块
def distance(x1,y1,x0,y0):
    return pow(pow(x1-x0,2)+pow(y1-y0,2),0.5)



song_list = []  # 歌曲列表
# 加载歌曲列表
def load_songs():
    songs = os.listdir("music")
    songs = [f"music/{song}" for song in songs]
    song_list.extend(songs)
    print(len(song_list))
    return(len(song_list))
    
# 播放指定的歌曲
def play(index):  # index当前歌曲的索引
    pygame.mixer.music.stop()
    # 播放音乐
    music = pygame.mixer.music.load(song_list[index])
    pygame.mixer.music.play()

pygame.mixer.init()

#摄像头读取函数
camera = cv2.VideoCapture(0)
#定义识别手部的对象
hand_detector  = HandDetector()
# #图片读取以及调整大小
# TNT_B = cv2.imread('TNT_B.jpg')
# x, y = TNT_B.shape[:2]
# TNT_B = cv2.resize(TNT_B, (int(y / 25), int(x / 25)))
# #daodan = cv2.flip(img2,0)

# TNT_G = cv2.imread('TNT_G.jpg')
# x, y = TNT_G.shape[:2]
# TNT_G = cv2.resize(TNT_G, (int(y / 25), int(x / 25)))
# #daodan = cv2.flip(img2,0)
#碰撞检测函数
def collision(finger_y,finger_x,aim_y,aim_x):
    game_model = 25
    if aim_x - game_model <= finger_x <= aim_x + game_model and aim_y - game_model <= finger_y <= aim_y + game_model:
        return True
        



def file_read(index):
    path1 = r"C:\Users\WX133\Documents\my_opencv\HandMusic\music"
    f = os.walk(path1)
    filenames_list = [filenames for dirpath, dirnames, filenames in f]
    #print(filenames_list)
    for i in filenames_list:
        name = str(i[index])[:-4]
        return name



def remove_file(old_path, new_path):
    print(old_path)
    print(new_path)
    filelist = os.listdir(old_path) #列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    print(filelist)
    for file in filelist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)
        print('src:', src)
        print('dst:', dst)
        shutil.move(src, dst)



bb = cv2.imread('playp.jpg')
x, y = bb.shape[:2]
bb = cv2.resize(bb, (int(y / 5), int(x / 5)))


# sourcery skip: avoid-builtin-shadow
next = cv2.imread('next.jpg')
x, y = next.shape[:2]
next = cv2.resize(next, (int(y / 5), int(x / 5)))

last = cv2.imread('last.jpg')
x, y = last.shape[:2]
last = cv2.resize(last, (int(y / 5), int(x / 5)))

# chang = cv2.imread('changpian.jpg')
# x, y = chang.shape[:2]
# chang = cv2.resize(chang, (int(y / 1), int(x / 1)))

add = cv2.imread('add.jpg')
x, y = add.shape[:2]
add = cv2.resize(add, (int(y / 5), int(x / 5)))

back = cv2.imread('back.jpg')
x, y = back.shape[:2]
back = cv2.resize(back, (int(y / 5), int(x / 5)))

delate = cv2.imread('delate.jpg')
x, y = delate.shape[:2]
delate = cv2.resize(delate, (int(y / 5), int(x / 5)))

black = cv2.imread('980.jpg')
x, y = black.shape[:2]
black = cv2.resize(black, (int(y / 1), int(x / 1)))

stop = cv2.imread('stop.jpg')
x, y = stop.shape[:2]
stop = cv2.resize(stop, (int(y / 5), int(x / 5)))







#初始化音乐播放
index = 0
mu = load_songs()

# 用来防止按钮过于灵敏

step = 0  
step_start = 0
step_delate = 0
step_stop = 0

button_model = True
button_start_model = True

button_model_last = True
button_model_right = True

button_model_delate = True
button_model_stop = True

viewtext = 0

music_model = True
music_stop_model = False

playmodel = True

reflex = 22

print(song_list)

#开始循环
while True:
    if playmodel:
        success, img = camera.read()  # 返回两个值，第一个是布尔类型反应有没有读取成功，img是每一帧的图片
        #h1, w1, c1 = img.shape
        if success:

            img = cv2.flip(img, 1) # 将画面翻转

            hand_detector.process(img)
            position = hand_detector.find_position(img)
            gameStop = not position['Left'] and not position['Right']
            #index = 0

            # path1 = r"C:\Users\WX133\Documents\my_opencv\HandMusic\music"
            # f = os.walk(path1)
            # print(path1)
            # if not pygame.mixer.music.get_busy():
            #     play(index)
            img = picplus(img,black,(0,0))

            

            if music_model:
                # img = image_plus(img,chang,(0,0))
                text_life = file_read(0)

                text_play = 'Music player'
                cv2.putText(img,text_play,(20,50),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,0),2)


                # root = tk.Tk()
                # if root.withdraw():

                #     f_path = filedialog.askopenfilename()
                #     print('\n获取的文件地址:', f_path)
                # else:
                #     continue


                #cv2.putText(img,text_life,(50,50),cv2.FONT_HERSHEY_PLAIN,2.0,(50,50,205),2)
                #cv2.circle(img, (380,200),20,(255, 255, 255),cv2.FILLED)

                img = image_plus(img,bb,(380-reflex,50-reflex))

                img = image_plus(img,add,(580-reflex-3,50-reflex-3))

                #cv2.circle(img, (400,400),10,(255, 255, 255),cv2.FILLED)


                if left_finger := position['Left'].get(8, None):

                    cv2.circle(img, (left_finger[0], left_finger[1]),10,(255, 0, 0),cv2.FILLED)
                    if button_start_model == True:  #如果按钮处于可以按的状态
                        if collision(left_finger[0],left_finger[1],380,50):


                            button_start_model = False
                            step_start = 30

                            button_model_stop = False
                            step_stop = 30


                            #if pygame.mixer.music.get_busy() == False:
                            play(index)
                            music_model = False
                            continue    
                    
                    else:
                        step_start = step_start - 1
                        if step_start == 0:
                            button_start_model = True 


                    cv2.circle(img, (left_finger[0], left_finger[1]),10,(255, 0, 0),cv2.FILLED)

                    if collision(left_finger[0],left_finger[1],580,50):
                        
                        playmodel = False
                        continue    


                if right_finger := position['Right'].get(8, None):

                    cv2.circle(img, (right_finger[0], right_finger[1]),10,(255, 0, 0),cv2.FILLED)
                    if button_start_model == True:  #如果按钮处于可以按的状态
                        if collision(right_finger[0],right_finger[1],380,50):


                            button_start_model = False
                            step_start = 30

                            button_model_stop = False
                            step_stop = 30


                            #if pygame.mixer.music.get_busy() == False:
                            play(0)
                            music_model = False
                            continue    
                    
                    else:
                        step_start = step_start - 1
                        if step_start == 0:
                            button_start_model = True 


                    cv2.circle(img, (right_finger[0], right_finger[1]),10,(255, 0, 0),cv2.FILLED)

                    if collision(right_finger[0],right_finger[1],580,50):
                        
                        playmodel = False
                        continue    

                

            else:
                
                text_life = file_read(index)
                cv2.putText(img,text_life,(20,50),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,0),2)


                #cv2.circle(img, (420,200),10,(255, 255, 255),cv2.FILLED)
                img = image_plus(img,next,(480-reflex,50-reflex))
                #cv2.circle(img, (420,400),10,(255, 255, 255),cv2.FILLED)
                img = image_plus(img,last,(280-reflex,50-reflex))

                img = image_plus(img,delate,(580-reflex,50-reflex))

                if pygame.mixer.music.get_busy():
                    img = image_plus(img,stop,(380-reflex,50-reflex))
                else:
                    img = image_plus(img,bb,(380-reflex,50-reflex))

                #cv2.circle(img, (440,50),10,(255, 255, 255),cv2.FILLED)

                #cv2.circle(img, (200,400),10,(0, 0, 0),cv2.FILLED)

                if left_finger := position['Left'].get(8, None):

                    cv2.circle(img, (left_finger[0], left_finger[1]),10,(255, 0, 0),cv2.FILLED)
                    if button_model == True:  #如果按钮处于可以按的状态
                        if collision(left_finger[0],left_finger[1],480,50):
                            pygame.mixer.music.stop()
                            if index == mu-1:
                                index = 0
                            else:
                                index = index + 1

                            print(index)
                            button_model = False
                            step = 30

                            if pygame.mixer.music.get_busy() == False:
                                play(index)
                            continue    
                    else:
                        step = step - 1
                        if step == 0:
                            button_model = True 




                    if button_model_last == True:  #如果按钮处于可以按的状态
                        if collision(left_finger[0],left_finger[1],280,50):
                            pygame.mixer.music.stop()
                            if index <=0:
                                index = mu - 1
                            else:
                                index = index - 1

                            print(index)
                            button_model_last = False
                            step_last = 30

                            if pygame.mixer.music.get_busy() == False:
                                play(index)
                            continue    
                    else:
                        step_last = step_last - 1
                        if step_last == 0:
                            button_model_last = True 



                    if button_model_delate == True:  #如果按钮处于可以按的状态
                        if collision(left_finger[0],left_finger[1],580,50):
                            text_life_delate = file_read(index)
                            pygame.mixer.music.stop()
                            if index == mu-1:
                                index = 0
                            else:
                                index = index + 1
                            play(index)


                            os.remove('music/{}.mp3'.format(text_life_delate))

                            song_list = []
                            mu = load_songs()

                            if index <=0:
                                index = mu - 1
                            else:
                                index = index - 1

                            play(index)
                            button_model_delate = False
                            step_delate = 30
                            continue    
                    else:
                        step_delate = step_delate - 1
                        if step_delate == 0:
                            button_model_delate = True 

                    
                    if button_model_stop == True:  #如果按钮处于可以按的状态
                        if collision(left_finger[0],left_finger[1],380,50):
                            if pygame.mixer.music.get_busy():
                                pygame.mixer.music.pause() #暂停音乐播放
                            else:
                                pygame.mixer.music.unpause() 

                            button_model_stop = False
                            step_stop = 30
                            continue    
                    else:
                        step_stop = step_stop - 1
                        if step_stop == 0:
                            button_model_stop = True 





                if right_finger := position['Right'].get(8, None):

                    cv2.circle(img, (right_finger[0], right_finger[1]),10,(255, 0, 0),cv2.FILLED)
                    if button_model == True:  #如果按钮处于可以按的状态
                        if collision(right_finger[0],right_finger[1],480,50):
                            pygame.mixer.music.stop()

                            if index == mu-1:
                                index = 0
                            else:
                                index = index + 1

                            print(index)

                            button_model = False
                            step = 30
                            if pygame.mixer.music.get_busy() == False:
                                play(index)
                            continue    
                    else:
                        step = step - 1
                        if step == 0:
                            button_model = True 




                    if button_model_last == True:  #如果按钮处于可以按的状态
                        if collision(right_finger[0],right_finger[1],280,50):
                            pygame.mixer.music.stop()

                            if index <=0:
                                index = mu - 1
                            else:
                                index = index - 1

                            print(index)

                            button_model_last = False
                            step_last = 30
                            if pygame.mixer.music.get_busy() == False:
                                play(index)
                            continue    
                    else:
                        step_last = step_last - 1
                        if step_last == 0:
                            button_model_last = True 



                    if button_model_delate == True:  #如果按钮处于可以按的状态
                        if collision(right_finger[0],right_finger[1],580,50):
                            text_life_delate = file_read(index)

                            pygame.mixer.music.stop()
                            if index == mu-1:
                                index = 0
                            else:
                                index = index + 1
                            play(index)


                            os.remove('music/{}.mp3'.format(text_life_delate))

                            song_list = []
                            mu = load_songs()

                            if index <=0:
                                index = mu - 1
                            else:
                                index = index - 1

                            play(index)
                            
                            button_model_delate = False
                            step_delate = 30
                            continue    
                    else:
                        step_delate = step_delate - 1
                        if step_delate == 0:
                            button_model_delate = True 

                    
                    if button_model_stop == True:  #如果按钮处于可以按的状态
                        if collision(right_finger[0],right_finger[1],380,50):
                            
                            if pygame.mixer.music.get_busy():
                                pygame.mixer.music.pause() #暂停音乐播放
                            else:
                                pygame.mixer.music.unpause() 

                            button_model_stop = False
                            step_stop = 30
                            continue    
                    else:
                        step_stop = step_stop - 1
                        if step_stop == 0:
                            button_model_stop = True 

            #if pygame.mixer.music.get_busy():
            if left_hand := position['Left'].get(0, None):
                if left_finger := position['Left'].get(8, None):
                    #判断手部是否握拳，只对左手有效
                    if left_hand[0] and left_hand[1] and left_finger[0] and left_finger[1] and distance(left_hand[0], left_hand[1], left_finger[0], left_finger[1]) <= 80:
                        pygame.mixer.music.pause() #暂停音乐播放
                        music_model = True

            if right_hand := position['Right'].get(0, None):
                if right_finger := position['Right'].get(8, None):
                    #判断手部是否握拳，只对左手有效
                    if right_hand[0] and right_hand[1] and right_finger[0] and right_finger[1] and distance(right_hand[0], right_hand[1], right_finger[0], right_finger[1]) <= 80:
                        pygame.mixer.music.pause() #暂停音乐播放
                        music_model = True


            #双手比X就退出画面
            if left_finger6 := position['Left'].get(6,None):
                if right_finger6:= position['Right'].get(6,None):
                    #print(distance(left_finger6[0],left_finger6[1],right_finger6[0],right_finger6[1]))
                    if distance(left_finger6[0],left_finger6[1],right_finger6[0],right_finger6[1]) <= 14:
                        break

            # elif left_hand := position['Left'].get(0, None):
            #     #判断手部是否松开，只对左手有效
            #     if left_hand[0] and left_hand[1] and left_finger[0] and left_finger[1] and distance(left_hand[0], left_hand[1], left_finger[0], left_finger[1]) >= 80:
            #         pygame.mixer.music.unpause() 
            


            if viewtext == 1:
                text2 = 'No more Music'
                cv2.putText(img,text2,(330,300),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2)

            

            cv2.imshow('video',img)

        #等待按键，等待的是1ms
        k = cv2.waitKey(1)

        # 如果k=q按键
        if k == ord('q'):  
            break
    else:
        root = tk.Tk()
        root.withdraw()
        f_path = filedialog.askopenfilename()
        
        print('\n获取的文件地址:', f_path)
        shutil.copy(str(f_path),'C:/Users/WX133/Documents/my_opencv/HandMusic/music')
        if f_path:
            song_list = []
            mu = load_songs()
            print(song_list)
            playmodel = True


#释放掉摄像头，并销毁窗口
camera.release()
cv2.destroyAllWindows()


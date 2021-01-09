import cv2, dlib, sys
import numpy as np

scaler = 0.1

detector = dlib.get_frontal_face_detector() # 얼굴 디렉터 모듈 초기화
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # 얼굴 특징점 모듈 초기화 

VideoFile = 'samples/girl.mp4'

cap = cv2.VideoCapture(VideoFile)#동영상 파일로드 
### cv2.VideoCapture(0) 비디오 대신 웹캠이 켜진다.

overlay = cv2.imread('samples/ryan_transparent.png',cv2.IMREAD_UNCHANGED)

# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
  bg_img = background_img.copy()
  # convert 3 channels to 4 channels
  if bg_img.shape[2] == 3:
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

  if overlay_size is not None:
    img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

  b, g, r, a = cv2.split(img_to_overlay_t)

  mask = cv2.medianBlur(a, 5)

  h, w, _ = img_to_overlay_t.shape
  roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

  img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
  img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

  bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

  # convert 4 channels to 4 channels
  bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

  return bg_img

face_roi = []
face_sizes = []

while True:
    ret, img = cap.read()#동영상 파일에서 frame 단위로 읽기 
    
    if not ret: # frame이 없다면 종료
        break
    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0]*scaler)))#video 크기 조절 
    ori = img.copy() #원본 

    # detect faces
    faces = detector(img)
    face = faces[0]


    dlib_shape = predictor(img, face) #img의 face영역안의 얼굴 특징점 찾기
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    #compute coneter and boundaries of face 
    top_left = np.min(shape_2d, axis=0)
    bottom_right = np.max(shape_2d, axis = 0)

    face_size = int(max(bottom_right - top_left)*1.5)

    #모든 특징점의 평균을 구해서 얼굴 중심 값을 구함
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)


    result = overlay_transparent(ori, overlay, center_x, center_y, overlay_size=(face_size, face_size))
    #visualize
    img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(),face.bottom()), color=(255,255,255),
                        thickness=  2, lineType=cv2.LINE_AA)

    # 얼굴 특징점의 갯수는 68개 
    # for loop를 이용해 68개를 cv2.circle()로 그린다
    for s in shape_2d:
        cv2.circle(img, center=tuple(s), radius = 1, color =(255,255,255),thickness=  2, lineType=cv2.LINE_AA)
    
    #왼쪽 상단, 오른쪽 하단에 파란색 점이 그려짐 
    cv2.circle(img, center = tuple(top_left),radius =1, color=(255,0,0), thickness = 2, lineType = cv2.LINE_AA)
    cv2.circle(img, center = tuple(bottom_right),radius =1, color=(255,0,0), thickness = 2, lineType = cv2.LINE_AA)
    cv2.circle(img, center = tuple((center_x,center_y)),radius =1, color=(0,0,255), thickness = 2, lineType = cv2.LINE_AA)
    
    cv2.imshow('img', img)
    cv2.imshow('result',  result)
    cv2.waitKey(1)#1ms 만큼 대기 




import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import time

import math
import re


model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
detection_result = None

tips_id = [4,8,12,16,20] #indice de los dedos que queremos detectar

landmarks_hist=[]
letters_hist=[]
dirs_hist=[]

max_hist=40

mov_letter_size=0
mov_letter=None
max_mov_letter_size=30

def update_hist(letter,lm):
    global landmarks_hist
    global letters_hist
    global dirs_hist
    global max_hist
    global tips_id

    x=lm[tips_id[1]].x
    y=lm[tips_id[1]].y

    if len(dirs_hist) == 0:
        dirs_hist.append('-')
    else:
        lm_ant=landmarks_hist[-1]
        x_ant=lm_ant[tips_id[1]].x
        y_ant=lm_ant[tips_id[1]].y

        if x_ant==x:
            dirs_hist.append('-')
        elif x_ant<x:
            dirs_hist.append('D')
        else:
            dirs_hist.append('I')


    letters_hist.append(letter) # Añadimos la letra al historial
    landmarks_hist.append(lm)

    if len(letters_hist) > max_hist:
          letters_hist.pop(0)
          landmarks_hist.pop(0)
          dirs_hist.pop(0)

def print_pos(detection_result):
  hand_landmarks_list = detection_result.hand_landmarks #list of landmarks, devuelve mas de una mano
  for lm in hand_landmarks_list:
    print(lm[tips_id[1]].x,lm[tips_id[1]].y,lm[tips_id[1]].z) #posicion del dedo indice

def print_angle(detection_result):
  hand_landmarks_list = detection_result.hand_landmarks #list of landmarks, devuelve mas de una mano
  for lm in hand_landmarks_list:
    x= lm[tips_id[1]].x - lm[tips_id[1]-3].x
    y= - lm[tips_id[1]].y + lm[tips_id[1]-3].y
    print(math.atan2(y,x)) #angulo en radianes

def print_distance(detection_result):
    hand_landmarks_list = detection_result.hand_landmarks #list of landmarks, devuelve mas de una mano
    for lm in hand_landmarks_list:
        x1= lm[tips_id[1]].x
        x2= lm[tips_id[1]-3].x
        y1=lm[tips_id[1]].y
        y2=lm[tips_id[1]-3].y
        print(math.sqrt((x2-x2)**2 + (y1-y2)**2)) #ditancia euclideana


def finger_infov2(lm):
    # Comparar 2 distancias a la base, abierto o cerrado para la mano entera
    global tips_id  # Cuando utilices una variable tips_id que está fuera de la función, tienes que declararla como global
    info = []  # Lista para almacenar la información de cada dedo

    for tip in tips_id:  # Para cada dedo, calculamos distancias
        x1 = lm[tip].x
        y1 = lm[tip].y
        x2 = lm[tip - 1].x
        y2 = lm[tip - 1].y
        x3 = lm[tip - 2].x
        y3 = lm[tip - 2].y
        x4 = lm[tip - 3].x
        y4 = lm[tip - 3].y
        x5 = lm[0].x
        y5 = lm[0].y

        # Calcula las distancias desde los puntos clave del dedo a la base de la mano
        d1 = math.sqrt((x1 - x5) ** 2 + (y1 - y5) ** 2)
        d2 = math.sqrt((x2 - x5) ** 2 + (y2 - y5) ** 2)
        d3 = math.sqrt((x3 - x5) ** 2 + (y3 - y5) ** 2)
        d4 = math.sqrt((x4 - x5) ** 2 + (y4 - y5) ** 2)

        # Encuentra la máxima distancia entre los puntos clave del dedo y la base de la mano
        max_d = max([d1, d2, d3, d4])
        extended = 0

        # Determina si el dedo está extendido comparando las distancias
        if d1 == max_d:
            extended = 1

        # Calcula el ángulo entre el dedo y la base de la mano
        ang = math.atan2(y4 - y1, x1 - x4) * 180 / np.pi

        #si la base del pulgar esta cerca de la base del dedo implica que no esta extendido
        if(tip==tips_id[0]):
            if(pulgar_superpuesto_sobre(lm,lm[20]) or pulgar_superpuesto_sobre(lm, lm[16]) or pulgar_superpuesto_sobre(lm, lm[12])):
                print("pulgar superpuesto sobre menique")
                extended=0
            if(pulgar_centro_palma(lm)):
                extended=0


        # Agrega la información del dedo a la lista
        info.append((extended, int(ang)))

    print(info)  # Imprime la información de todos los dedos
    return info  # Devuelve la información de los dedos en forma de lista de tuplas


def pulgar_superpuesto_sobre(lm, dedo):
    # Verifica si el dedo medio está superpuesto al dedo índice
    base_pulgar = lm[4]

    # Calcula la distancia entre los puntos clave de los dedos
    d = math.sqrt((base_pulgar.x - dedo.x) ** 2 + (base_pulgar.y - dedo.y) ** 2)

    if d < 0.1:
        return True


def pulgar_centro_palma(lm):
    base_pulgar = lm[4]
    inicio_indice = lm[5]
    inicio_medio = lm[9]
    inicio_anular = lm[13]
    inicio_menique = lm[17]
    inicio_indice = lm[5]
    centro_palma = lm[0]


    # Calcula el área total del triángulo formado por los puntos 0, 5 y 17
    area_total = abs(
        (inicio_menique.x * (centro_palma.y - inicio_indice.y) +
         centro_palma.x * (inicio_indice.y - inicio_menique.y) +
         inicio_indice.x * (inicio_menique.y - centro_palma.y)) / 2
    )

    # Calcula el área de los tres subtriángulos formados al agregar el punto 4
    area1 = abs(
        (base_pulgar.x * (centro_palma.y - inicio_indice.y) +
         centro_palma.x * (inicio_indice.y - base_pulgar.y) +
         inicio_indice.x * (base_pulgar.y - centro_palma.y)) / 2
    )

    area2 = abs(
        (inicio_menique.x * (base_pulgar.y - centro_palma.y) +
         base_pulgar.x * (centro_palma.y - inicio_menique.y) +
         centro_palma.x * (inicio_menique.y - base_pulgar.y)) / 2
    )

    area3 = abs(
        (inicio_indice.x * (base_pulgar.y - inicio_menique.y) +
         base_pulgar.x * (inicio_menique.y - inicio_indice.y) +
         inicio_menique.x * (inicio_indice.y - base_pulgar.y)) / 2
    )

    # Si la suma de las áreas de los subtriángulos es igual al área total, entonces el punto está dentro del triángulo
    if area1 + area2 + area3 == area_total:
        print("centro palma")
        return True
    else:
        return False





def draw_bb_with_letter(image,letter,size):

  font = cv2.FONT_HERSHEY_SIMPLEX
  font_size = 3
  font_color = (255,255,255) #BGR
  font_thickness = 3

  h, w, _ = image.shape
  text_size = cv2.getTextSize(letter, font, font_size, font_thickness)[0]
  text_w, text_h = text_size
  cv2.putText(image, letter, (int((w - text_w) / 2), int((h + text_h) / 2)), font, font_size, font_color, font_thickness)

  return image

def draw_bb_with_letter(image,detection_result,letter):
  
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_size = 3
  font_color = (255,255,255) #BGR
  font_thickness = 3

  bb_color = (0,255,0)
  margin = 10
  bb_thickness = 3
  # Loop through the detected hands to visualize.
  hand_landmarks_list = detection_result.hand_landmarks
  for hand_landmarks in hand_landmarks_list:
    
    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    min_x = int(min(x_coordinates) * width) - margin
    min_y = int(min(y_coordinates) * height) - margin
    max_x = int(max(x_coordinates) * width) + margin
    max_y = int(max(y_coordinates) * height) + margin

    # Draw a bounding-box
    cv2.rectangle(image, (min_x,min_y),(max_x,max_y),bb_color,bb_thickness)

    # Get the text size
    text_size, _ = cv2.getTextSize(letter, font, font_size, font_thickness)
    text_w, text_h = text_size
    # Draw background filled rectangle
    cv2.rectangle(image, (min_x,min_y), (min_x + text_w, min_y - text_h), bb_color, -1)  
    # Draw the letter
    cv2.putText(image, letter,(min_x, min_y), font, font_size, font_color, font_thickness, cv2.LINE_AA)
  
  return image



def moving_letter(letter, pattern=r'D{5,}I{5,}D{5,}'):
  directions = ''.join(dirs_hist)
  directions= re.sub(r'D-D', 'DD', directions)
  directions= re.sub(r'D-I', 'DI', directions)
  directions= re.sub(r'I-D', 'ID', directions)
  directions= re.sub(r'I-I', 'II', directions)
  print(directions)
  matches = re.findall(pattern, directions)

  #comprobar letra
  letters = ''.join(letters_hist)
  pattern2 = r'('+letter+'){20,}'
  matches2 = re.findall(pattern2, letters)

  if len(matches) and len(matches2):
    return True
  return False



def get_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
  global detection_result
  detection_result = result

def draw_landmarks_on_image(rgb_image, detection_result):

  hand_landmarks_list = detection_result.hand_landmarks #list of landmarks, devuelve mas de una mano
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for hand_landmarks in hand_landmarks_list:
 
    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

  return annotated_image

def draw_finger_info(letter, lm, image):
    global tips_id
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    font_thickness = 2
    margin = 10

    # Supongamos que deseas dibujar la letra en la posición del dedo índice
    h, w, _ = image.shape
    x = int(lm[tips_id[1]].x * w)
    y = int(lm[tips_id[1]].y * h) - margin

    font_color = (0, 255, 0)
    cv2.putText(image, letter, (x, y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

    return image

def map_to_letter(finger_info_result,lm):
    # Define el mapeo de resultados de finger_info a letras
    letter_mapping = {
        (1, 0, 0, 0, 0): 'E', #ok
        (1, 1, 0, 0, 0): 'L', #ok
        (0, 1, 1, 1, 0): 'P', #ok
        (0, 0, 0, 0, 0): 'A', #ok
        (0, 0, 0, 0, 1): 'I', #mas o menos
        (1, 0, 1, 1, 1): 'F', #ok
        (0, 1, 1, 0, 0): 'U', #ok
    }

    # Obtener solo los primeros elementos de la tupla
    finger_info_result_clean = tuple(element[0] for element in finger_info_result)

    # Buscar el resultado en el mapeo y devolver la letra correspondiente
    for pattern, letter in letter_mapping.items():
        # Verifica si el resultado coincide con el patrón (sin tener en  cuenta el ángulo)
        if finger_info_result_clean == pattern:
            if letter == 'U' and finger_info_result[1][1] < -80 and finger_info_result[2][1] < -80:
                update_hist('N',lm)
                return 'N' #ok pero mejorar la N
            if letter=='P' and finger_info_result[2][1] < -80 and finger_info_result[3][1] < -80:
                update_hist('M',lm)
                return 'M' #ok
            if letter=='U' and letra_R(lm):
                update_hist('R',lm)
                return 'R' #ok

            update_hist(letter,lm)
            return letter


    return 'No se encontró coincidencia'


def letra_R(lm):
    # Verifica si el dedo medio está superpuesto al dedo índice
    base_indice = lm[8]  # Base del dedo índice
    base_medio = lm[12]   # Base del dedo medio
    print(base_medio.x)
    print(base_indice.x)
    #si la base del dedo medio esta a la izquierda de la base del dedo indice
    if base_medio.x <= base_indice.x:
        return True
    else:
        return False


#--------------------------------------------------------------------------------------------------------------------------

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=get_result)

with HandLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
  cap = cv2.VideoCapture(0)
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    image = cv2.flip(image,1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    frame_timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, frame_timestamp_ms)
    if detection_result is not None:
      image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
      #print_pos(detection_result)
      #print_angle(detection_result)

      if(len(detection_result.hand_landmarks) > 0):
        lm= detection_result.hand_landmarks[0]
        info=finger_infov2(lm)
        letter=map_to_letter(info,lm)
        image = draw_bb_with_letter(image,detection_result,letter)


      if mov_letter_size == 0:
          if moving_letter("L",  r'D{5,}I{5,}D{5,}|I{5,}D{5,},I{5,}'):
            letter = "LL" #ok
            image = draw_bb_with_letter(image,detection_result,letter)

          if moving_letter("R", r'D{3,}I{3,}D{3,}|I{3,}D{3,},I{3,}'):
            letter = "RR"
            image = draw_bb_with_letter(image,detection_result,letter)

      if moving_letter("U",  r'D{3,}I{3,}D{3,}|I{3,}D{3,},I{3,}'):
            letter = "V" #ok
            image = draw_bb_with_letter(image,detection_result,letter)

      if moving_letter("P", r'D{3,}I{3,}D{3,}|I{3,}D{3,},I{3,}'):
            letter = "W"
            image = draw_bb_with_letter(image,detection_result,letter)

      if moving_letter("I", r'D{3,}I{3,}D{3,}|I{3,}D{3,},I{3,}'):
            letter = "J" #ok
            image = draw_bb_with_letter(image,detection_result,letter)

      if moving_letter("N", r'D{3,}I{3,}D{3,}|I{3,}D{3,},I{3,}'):
            letter = "eñe"
            image = draw_bb_with_letter(image,detection_result,letter)


    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break




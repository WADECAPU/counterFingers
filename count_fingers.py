import cv2
import mediapipe as mp  
import numpy as np 
from math import acos, degrees


def palm_centroid(coordinates_list):
    coordinates = np.array(coordinates_list)
    centroid = np.mean(coordinates, axis=0) # Se sumaran todas las coordenadas en X y se dividan con el numero de elementos x y de y
    centroid = int(centroid[0]), int(centroid[1])
    return centroid

np_drawing = mp.solutions.drawing_utils # drawing y styles, para poder visualizar las conexiones y lankmarks ya que
np_drawing_styles = mp.solutions.drawing_styles 
np_hands = mp.solutions.hands # Porque vamos a realizar un programa que lea los dedos de la mano

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # 

# Puntos o coordenadas que forman el Pulgar
thumb_points = [1,2,4]

# Indice, medio, anular y meñique
#Palma de la mano
palm_points = [0,1,2,5,9,13,17]
# Punta de cada dedo
fingertips_points = [8,12,16, 20]
finger_base_points = [6, 10, 14 , 18] 

# Colores de los cuadros de cada dedo
GREEN = (48, 255, 48)
BLUE = (192, 101, 21)
YELLOW = (0, 204, 255)
PURPLE = (128, 64, 128)
PEACH = (180, 229, 255)

# Algunas especificaciones
with np_hands.Hands(
    model_complexity = 1,
    max_num_hands = 1, # Maximo de manos a detectar
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as hands:

    while True:
        ret, frame = cap.read() # Leer los fotogrmas que nos da nuestra camara
        if ret == False:
            break
        frame = cv2.flip(frame, 1) # El flip es para que esos fotogramas de camara no refleje como si fuera un espejo
        height, width, _ = frame.shape # Alto y ancho 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # La lectura de una imagen 
        results = hands.process(frame_rgb)
        fingers_counter = '_'
        thickness = [2, 2, 2, 2, 2] # Para el relleno de los cuadros de los dedos

        if results.multi_hand_landmarks: # Para saber si la deteccion exite 
            coordinates_thumb = [] # Listas vacias para almecenar los coordenas 
            coordinates_palm = []
            coordinates_ft = []
            coordinates_fb = []
            for hand_landmarks in results.multi_hand_landmarks:
                for index in thumb_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_thumb.append([x,y])

                for index in palm_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_palm.append([x,y])

                for index in fingertips_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_ft.append([x,y])

                for index in finger_base_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_fb.append([x,y])

                #Puntos ubicados del Pulgar
                p1 = np.array(coordinates_thumb[0])
                p2 = np.array(coordinates_thumb[1])
                p3 = np.array(coordinates_thumb[2])

                # Para calcular la distancia mediante lineas de los puntos
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)

                # El angulo
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 *l3)))
                # print(angle)
                thumb_finger = np.array(False)
                if angle > 150:
                    thumb_finger = np.array(True)

                # Indice, medio, anular y meñique
                nx, ny = palm_centroid(coordinates_palm)
                cv2.circle(frame, (nx, ny), 3, (0,  255, 0), 2)
                coordinates_centroid = np.array([nx, ny])
                coordinates_ft = np.array(coordinates_ft)
                coordinates_fb = np.array(coordinates_fb)

                # Calcular las distancias 
                d_centroid_ft = np.linalg.norm(coordinates_centroid - coordinates_ft, axis=1) # Axis sirve para la identificacion de cada dedo
                d_centroid_fb = np.linalg.norm(coordinates_centroid - coordinates_fb, axis=1)
                dif = d_centroid_ft - d_centroid_fb
                fingers = dif > 0 # Si esta extendido o no los 4 dedos
                fingers = np.append(thumb_finger, fingers) # Para ver si ya los 5 dedos estan extendidos o no
                fingers_counter = str(np.count_nonzero(fingers==True)) # Ahora para decir el numero del dedo levantado

                for (i, finger) in enumerate(fingers):
                    if finger == True:
                        thickness[i] = -1

                np_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks,
                    np_hands.HAND_CONNECTIONS,
                    np_drawing_styles.get_default_hand_landmarks_style(),
                    np_drawing_styles.get_default_hand_connections_style(),
                )

        # Visulaizacion de fingers_counter, es decir una mano
        cv2.rectangle(frame, (0,0), (80,80), (125, 220, 0), -1)
        cv2.putText(frame, fingers_counter, (15, 65), 1, 5, (255, 255, 255), 2)
        # Cuadro para pulgar
        cv2.rectangle(frame, (100,10), (150,60), PEACH, thickness[0])
        cv2.putText(frame, 'Pulgar', (100, 80), 1, 1, (179, 182, 183), 2)

        # Cuadro para el Indice
        cv2.rectangle(frame, (160,10), (210,60), PURPLE, thickness[1])
        cv2.putText(frame, 'Indice', (160, 80), 1, 1, (255, 255, 255), 2)

        # Medio 
        cv2.rectangle(frame, (220,10), (270,60), YELLOW, thickness[2])
        cv2.putText(frame, 'Medio', (220, 80), 1, 1, (255, 255, 255), 2)

        #Anular
        cv2.rectangle(frame, (280,10), (330,60), GREEN, thickness[3])
        cv2.putText(frame, 'Anular', (280, 80), 1, 1, (255, 255, 255), 2)

        # Meñique
        cv2.rectangle(frame, (340,10), (390,60), BLUE, thickness[4])
        cv2.putText(frame, 'Menique', (340, 80), 1, 1, (255, 255, 255), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


cap.release()
cv2.destroyAllWindows()

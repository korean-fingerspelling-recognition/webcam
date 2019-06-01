"""
This module is the main module in this package. It loads emotion recognition model from a file,
shows a webcam image, recognizes face and it's emotion and draw emotion on the image.
"""
from cv2 import WINDOW_NORMAL

import cv2
from face_detect import find_faces
import numpy as np
from PIL import Image
import time

x = 0
y = 0
h = 1200
w = 720

def nparray_as_image(nparray, mode='RGB'):
    return Image.fromarray(np.asarray(np.clip(nparray, 0, 255), dtype='uint8'), mode)

def _load_emoticons(images):
    return [nparray_as_image(cv2.imread('graphics/%s.png' % image, -1), mode=None) for image in images]

def _load_hangles(images):
    return [nparray_as_image(cv2.imread('spellings/%s.png' % spelling, -1), mode=None) for spelling in spellings]

def image_as_nparray(image):
    return np.asarray(image)

def hangle_list_modify(prediction_hangle, list):

    if ( len(list) != 0  ) :
        #if gesture = del
        if ( prediction_hangle != list[len(list)-1] ):
            if (prediction_hangle == 0) :
                del list[len(list)-1]

            elif (prediction_hangle != 0) :
                list.append(prediction_hangle)
    else :
        list.append(prediction_hangle)

    return list

def hangle_list_to_image(list):
    list_1_sheep = [5, 7, 5]
    list_2_cat = [1, 8, 5, 7, 5, 5, 10]
    list_3_dog = [1, 11]
    list_4_duck = [5, 8, 3, 10]
    list_5_giraffe = [1, 11, 3, 10, 2]
    list_6_rain = [4, 10]
    list_7_sun = [6, 11]
    list_8_snow = [2, 9, 2]

    list_tot =  [list_1_sheep, list_2_cat, list_3_dog, list_4_duck, list_5_giraffe,
                list_6_rain, list_7_sun, list_8_snow]

    prediction = 100

    for i in list_tot :
        if list == i :
            prediction = list_tot.index(i)
            break

    return prediction

def draw_with_alpha(source_image, image_to_draw, coordinates):
    x, y, w, h = coordinates
    image_to_draw = image_to_draw.resize((w, h), Image.ANTIALIAS)
    image_array = image_as_nparray(image_to_draw)
    for c in range(0, 3):
        source_image[y:y + h, x:x + w, c] = image_array[:, :, c] * (image_array[:, :, 3] / 255.0) \
                                            + source_image[y:y + h, x:x + w, c] * (1.0 - image_array[:, :, 3] / 255.0)


if __name__ == '__main__':
    images = ['1_sheep','2_cat', '3_dog', '4_duck', '5_giraffe', '6_rain', '7_sun', '8_snow', '9_board']
    spellings = ['0_del','1_g', '2_n', '3_l', '4_b', '5_o', '6_h', '7_ya', '8_oh', '9_woo', '10_ee', '11_a']
    emoticons = _load_emoticons(images)
    hangles = _load_hangles(spellings)

    # load model
    #fisher_face = cv2.face.FisherFaceRecognizer_create()
    #fisher_face.read('models/emotions_detection_model.xml')
    # use learnt model

    window_name = 'WEBCAM (press ESC to exit)'
    window_size = (1200, 720)
    window_name = window_name
    update_time = 8
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        read_value, webcam_image = vc.read()
    else:
        print("webcam not found")

    flag = 0
    list = []
    a = 1

    while read_value:

        #prediction = model.predict(normalized_face)  # do prediction
        list = [1, 8, 5, 7, 5, 5, 10]
        prediction_hangle = 10

        list = hangle_list_modify(prediction_hangle, list)
        prediction = hangle_list_to_image(list)

        if (prediction != 100 and a == 1) :

            if ( flag == 0 ):
                Max_Time = time.time() + 5
                flag = 1

            while ( Max_Time >= time.time() ):
                image_to_draw = emoticons[prediction]

                image_to_draw = image_to_draw.resize((h, w), Image.ANTIALIAS)
                image_array = image_as_nparray(image_to_draw)

                for c in range(0, 3):
                    webcam_image[y:y + 720, x:x + 1200, c] = image_array[:, :, c] * (image_array[:, :, 3] / 255.0) \
                                                                + webcam_image[y:y + 720, x:x + 1200, c] * (1.0 - image_array[:, :, 3] / 255.0)

                cv2.imshow(window_name, webcam_image)
                read_value, webcam_image = vc.read()
                key = cv2.waitKey(update_time)

            list = []
            a = 0

        else :
            image_board = emoticons[8]
            draw_with_alpha(webcam_image, image_board, (10,500,850,180))

            idx = 0
            for i in list :
                draw_with_alpha(webcam_image, hangles[i], (80 + idx*100,520,100,120))
                idx = idx + 1

        cv2.imshow(window_name, webcam_image)
        read_value, webcam_image = vc.read()
        key = cv2.waitKey(update_time)
        flag = 0
        print(2)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow(window_name)

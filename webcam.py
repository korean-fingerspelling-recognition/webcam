from cv2 import WINDOW_NORMAL
import cv2
import numpy as np
import time
import tensorflow as tf
import parameters as par
from PIL import ImageOps, Image
import operator

x = 0
y = 0
h = 1200
w = 720
image_size = 48
dim = 4

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

            elif (prediction_hangle != 0 and len(list) < 7) :
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

def convert_img(frame):

    flags = tf.app.flags
    FLAGS = flags.FLAGS

    frame = cv2.resize(frame, (image_size, image_size))
    convert_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_boundary = np.array([0, 40, 30], dtype="uint8")
    high_boundary = np.array([43, 255, 254], dtype = "uint8")

    skinmask = cv2.inRange(convert_img, low_boundary, high_boundary)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skinmask = cv2.erode(skinmask, kernel, iterations = 2)
    skinmask = cv2.dilate(skinmask, kernel, iterations = 2)

    low_boundary_2 = np.array([170, 80, 30], dtype="uint8")
    high_boundary_2 = np.array([180, 255, 250], dtype="uint8")

    skinmask2 = cv2.inRange(convert_img, low_boundary_2, high_boundary_2)
    skinmask = cv2.addWeighted(skinmask, 0.5, skinmask2, 0.5, 0.0)
    skinmask = cv2.medianBlur(skinmask, 5)
    skin = cv2.bitwise_and(frame, frame, mask=skinmask)
    frame = cv2.addWeighted(frame, 1.5, skin, -0.5, 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinmask)
    h, s, v = cv2.split(skin)
    return v # black except the part with hand


if __name__ == '__main__':
    images = ['1_sheep','2_cat', '3_dog', '4_duck', '5_giraffe', '6_rain', '7_sun', '8_snow', '9_board']
    spellings = ['0_del','1_g', '2_n', '3_l', '4_b', '5_o', '6_h', '7_ya', '8_oh', '9_woo', '10_ee', '11_a']
    emoticons = _load_emoticons(images)
    hangles = _load_hangles(spellings)

    window_name = 'WEBCAM (press ESC to exit)'
    window_size = (1200, 720)
    window_name = window_name
    update_time = 8
    cv2.namedWindow(window_name, WINDOW_NORMAL)

    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    vc = cv2.VideoCapture(0)

    sess = tf.Session()
    saver_gesture = tf.train.import_meta_graph('./Saved/' + str('501.meta'))

    saver_gesture.restore(sess, tf.train.latest_checkpoint('./Saved/'))

    # Get Operations to restore
    graph_gesture = sess.graph

    # Get Input Graph
    X = graph_gesture.get_tensor_by_name('Input:0')
    #Y = graph.get_tensor_by_name('Target:0')
    # keep_prob = tf.placeholder(tf.float32)
    #keep_prob = graph_gesture.get_tensor_by_name('Placeholder:0')

    # Get Ops
    prediction = graph_gesture.get_tensor_by_name('prediction:0')
    #logits = graph_gesture.get_tensor_by_name('logits:0')
    accuracy = graph_gesture.get_tensor_by_name('accuracy:0')

    if vc.isOpened():
        read_value, webcam_image = vc.read()
    else:
        print("webcam not found")

    flag = 0
    list = []
    a = 1

    while read_value:

        '''
        hand_image = object_detecion(webcam_image)
        '''
        hand_image = cv2.resize( webcam_image, (image_size, image_size))
        convert = convert_img(webcam_image)
        #mask_images.append(convert)
        hand_image = np.reshape(hand_image, [-1, image_size, image_size, 3])
        convert = np.reshape(convert, [-1, image_size, image_size, 1])
        testImage = np.concatenate((hand_image, convert), axis = 3)

        testImage = np.reshape(testImage, [-1, image_size, image_size, dim])
        testImage = testImage.astype(np.float32)
        testY = sess.run(prediction, feed_dict={X: testImage})
        prediction_hangle = int(np.argmax(testY));

        print(prediction_hangle)
        print("list" + str(list))

        list = hangle_list_modify(prediction_hangle, list)
        prediction_word = hangle_list_to_image(list)

        if (prediction_word != 100) :

            if ( flag == 0 ):
                Max_Time = time.time() + 5
                flag = 1

            while ( Max_Time >= time.time() ):
                image_to_draw = emoticons[prediction_word]

                image_to_draw = image_to_draw.resize((h, w), Image.ANTIALIAS)
                image_array = image_as_nparray(image_to_draw)

                for c in range(0, 3):
                    webcam_image[y:y + 720, x:x + 1200, c] = image_array[:, :, c] * (image_array[:, :, 3] / 255.0) \
                                                                + webcam_image[y:y + 720, x:x + 1200, c] * (1.0 - image_array[:, :, 3] / 255.0)

                cv2.imshow(window_name, webcam_image)
                read_value, webcam_image = vc.read()
                key = cv2.waitKey(update_time)

            list = []

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
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow(window_name)

from PIL import Image
import cv2
import numpy as np
import glob
import os
from autocrop.autocrop import check_underexposed, bgr_to_rbg

count = 0
for file in glob.iglob(r'PATH/TO/DATASET/CAER-S/train_processed/*_failed/*.png', recursive=True):
    count += 1
    print('{0:5d}/{1:5d}'.format(count, len(glob.glob(r'PATH/TO/DATASET/CAER-S/train_preprocessed/*_failed/*.png'))))
    print(file)

    new_file = file.split('/')
    new_file[-2] = new_file[-2][:-7]
    if not os.path.exists('/'.join(new_file[:-1])):
        os.makedirs('/'.join(new_file[:-1]))

    cropping = False

    x_start, y_start, x_end, y_end = 0, 0, 0, 0

    image = cv2.imread(file)
    oriImage = image.copy()


    def mouse_crop(event, x, y, flags, param):
        # grab references to the global variables
        global x_start, y_start, x_end, y_end, cropping

        # if the left mouse button was DOWN, start RECORDING
        # (x, y) coordinates and indicate that cropping is being
        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True

        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping == True:
                x_end, y_end = x, y

        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates
            x_end, y_end = x, y
            cropping = False  # cropping is finished

            refPoint = [(x_start, y_start), (x_end, y_end)]

            if len(refPoint) == 2:  # when two points were found
                roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                cv2.imshow("Cropped", roi)

                roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
                roi = check_underexposed(roi, cv2.cvtColor(oriImage, cv2.COLOR_BGR2GRAY))
                Image.fromarray(bgr_to_rbg(roi)).save(
                    os.path.join('/'.join(new_file[:-1]), file[-8:-4] + '_cropped.png')
                )

                mask = np.ones_like(oriImage)
                mask[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]] = 0
                image_masked = oriImage * mask
                Image.fromarray(bgr_to_rbg(image_masked)).save(
                    os.path.join('/'.join(new_file[:-1]), file[-8:-4] + '_masked.png')
                )

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)

    while True:

        i = image.copy()

        if not cropping:
            cv2.imshow("image", image)

        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)

        if cv2.waitKey(1) & 0xFF == 13:
            break

# close all open windows
cv2.destroyAllWindows()

count = 0
for file in glob.iglob(r'PATH/TO/DATASET/CAER-S/test_processed/*_failed/*.png', recursive=True):
    count += 1
    print('{0:5d}/{1:5d}'.format(count, len(glob.glob(r'PATH/TO/DATASET/CAER-S/test_preprocessed/*_failed/*.png'))))
    print(file)

    new_file = file.split('/')
    new_file[-2] = new_file[-2][:-7]
    if not os.path.exists('/'.join(new_file[:-1])):
        os.makedirs('/'.join(new_file[:-1]))

    cropping = False

    x_start, y_start, x_end, y_end = 0, 0, 0, 0

    image = cv2.imread(file)
    oriImage = image.copy()


    def mouse_crop(event, x, y, flags, param):
        # grab references to the global variables
        global x_start, y_start, x_end, y_end, cropping

        # if the left mouse button was DOWN, start RECORDING
        # (x, y) coordinates and indicate that cropping is being
        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True

        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping == True:
                x_end, y_end = x, y

        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates
            x_end, y_end = x, y
            cropping = False  # cropping is finished

            refPoint = [(x_start, y_start), (x_end, y_end)]

            if len(refPoint) == 2:  # when two points were found
                roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                cv2.imshow("Cropped", roi)

                roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
                roi = check_underexposed(roi, cv2.cvtColor(oriImage, cv2.COLOR_BGR2GRAY))
                Image.fromarray(bgr_to_rbg(roi)).save(
                    os.path.join('/'.join(new_file[:-1]), file[-8:-4] + '_cropped.png')
                )

                mask = np.ones_like(oriImage)
                mask[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]] = 0
                image_masked = oriImage * mask
                Image.fromarray(bgr_to_rbg(image_masked)).save(
                    os.path.join('/'.join(new_file[:-1]), file[-8:-4] + '_masked.png')
                )

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)

    while True:

        i = image.copy()

        if not cropping:
            cv2.imshow("image", image)

        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)

        if cv2.waitKey(1) & 0xFF == 13:
            break

# close all open windows
cv2.destroyAllWindows()



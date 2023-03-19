import cv2 as cv
import numpy as np
import os


DIR = r'/Users/anilbudak/VSCode/image_processing/images'
DIR_TRANS = r'/Users/anilbudak/VSCode/image_processing/test/data/'
#DIR_TRANS2 = r'/Users/anilbudak/VSCode/image_processing/multiplied_data/translated1/'
#DIR_TRANS3 = r'/Users/anilbudak/VSCode/image_processing/multiplied_data/translated2/'
#DIR_TRANS4 = r'/Users/anilbudak/VSCode/image_processing/multiplied_data/translated3/'
#DIR_TRANS5 = r'/Users/anilbudak/VSCode/image_processing/multiplied_data/translated4/'
#DIR_TRANS6 = r'/Users/anilbudak/VSCode/image_processing/multiplied_data/translated5/'

blank = np.zeros((32, 32, 3), np.uint8)

name = "0"

for images in os.listdir(DIR):
    image_path = os.path.join(DIR, images)
    image = cv.imread(image_path)
    for i in range(-9, 8):
        translation_matrix1 = np.float32([[1, 0, i/5], [0, 1, 0]])
        img_translation1 = cv.warpAffine(image, translation_matrix1, (32, 32))
        cv.imwrite(DIR_TRANS + name + '.jpg', img_translation1)
        name = str(int(name) + 1)
        #print(name + '.jpg')

        x = 0.6 + i/40
        result1 = cv.addWeighted(img_translation1, x, blank, 1-x, 0)
        cv.imwrite(DIR_TRANS + name + '.jpg', result1)
        name = str(int(name) + 1)


        translation_matrix2 = np.float32([[1, 0, 0], [0, 1, i/5]])
        img_translation2 = cv.warpAffine(image, translation_matrix2, (32, 32))
        cv.imwrite(DIR_TRANS + name + '.jpg', img_translation2)
        name = str(int(name) + 1)

        x = 0.6 + i / 40
        result2 = cv.addWeighted(img_translation2, x, blank, 1 - x, 0)
        cv.imwrite(DIR_TRANS + name + '.jpg', result2)
        name = str(int(name) + 1)


        translation_matrix3 = np.float32([[1, 0, i/5], [0, 1, i/5]])
        img_translation3 = cv.warpAffine(image, translation_matrix3, (32, 32))
        cv.imwrite(DIR_TRANS + name + '.jpg', img_translation3)
        name = str(int(name) + 1)

        x = 0.6 + i / 40
        result3 = cv.addWeighted(img_translation3, x, blank, 1 - x, 0)
        cv.imwrite(DIR_TRANS + name + '.jpg', result1)
        name = str(int(name) + 1)

#DEBUG
print('SUCCESS')
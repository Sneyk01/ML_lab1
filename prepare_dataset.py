"""
This script is used to create more diverse dataset and prepare data annotations for further recognition or training
Parameters:
    - NEED_ROTATE: True if you want to create new images in your dataset (rotate origin images)
    - IMG_COUNT: Numb of original images
    - ROOT_PATH: Path to main folder of dataset
    - FOLDERS: Names of class-folders

Struct of dataset repo:
    -ROOT_PATH
    |--FOLDERS[0]
      |--i0_0.png (original image)
      |--i0_5.png (copy rotated 5 degrees)
      |...
      |--i1_0.png
      |...
    |--FOLDER[1]
    |...
"""

from PIL import Image
from numpy import asarray

NEED_ROTATE = True
IMG_COUNT = 21
ROOT_PATH = 'dataSet/Learning'
FOLDERS = ["Photoshop", "Canva", "Corel", "Dreamweaver", "ExperienceDesign", "Flash", "Illustrator", "InDesign",
           "Pixlr", "Vectr"]

white = (255, 255, 255)
angels = [0, 5, 10, 15, 20, -5, -10, -15, -20]


# rotate part (create new image with diff rotate)
if NEED_ROTATE:
    for folder in FOLDERS:
        for img_index in range(IMG_COUNT):
            img_path = f'{ROOT_PATH}/{folder}/i{img_index}_0.png'
            try:
                im = Image.open(img_path)

                for angel in angels:
                    im_rotate = im.rotate(angel, fillcolor=white)
                    im_rotate.save(f'{ROOT_PATH}/{folder}/i{img_index}_{angel}.png', quality=100)

                im.close()
            except:
                print(f'Can`t open: {img_path}')

# .csv part (create annotation file for all img in root path)
img_id = 0
csv_annotation = open(f'{ROOT_PATH}/annotation.csv', 'w')
for folder in FOLDERS:
    for img_index in range(IMG_COUNT):
        for angel in angels:
            img_path = f'{ROOT_PATH}/{folder}/i{img_index}_{angel}.png'
            print(f'Open: {img_path}')
            try:
                im = Image.open(img_path)
                pixels = asarray(im)
                bin_img = ""
                for i in range(im.height):
                    for px in pixels[i]:
                        if px[0] == 255 and px[1] == 255 and px[2] == 255:
                            bin_img += '0; '
                        else:
                            bin_img += '1; '

                bin_img += f'{folder}; {img_id}; \r'
                csv_annotation.write(bin_img)
                img_id += 1

                im.close()

            except:
                print(f'Can`t open: {img_path}')

csv_annotation.close()

from PIL import Image
from numpy import asarray

need_rotate = True
img_count = 12
white = (255, 255, 255)
angels = [0, 5, 10, 15, 20, -5, -10, -15, -20]
folders = ["Photoshop", "Canva", "Corel", "Dreamweaver", "ExperienceDesign", "Flash", "Illustrator", "InDesign",
           "Pixlr", "Vectr"]

# rotate part
if need_rotate:
    for folder in folders:
        for img_index in range(img_count):
            im = Image.open(f'dataSet/{folder}/i{img_index}.png')

            for angel in angels:
                im_rotate = im.rotate(angel, fillcolor=white)
                im_rotate.save(f'dataSet/{folder}/i{img_index}_{angel}.png', quality=100)

            im.close()

# .csv part
id = 0
csv_annotation = open('dataSet/annotation.csv', 'w')
for folder in folders:
    for img_index in range(img_count):
        for angel in angels:
            print(f'Open: dataSet/{folder}/i{img_index}_{angel}.png')
            im = Image.open(f'dataSet/{folder}/i{img_index}_{angel}.png')
            pixels = asarray(im)
            bin_img = ""
            for i in range(im.height):
                for px in pixels[i]:
                    if px[0] == 255 and px[1] == 255 and px[2] == 255:
                        bin_img += '0; '
                    else:
                        bin_img += '1; '

            bin_img += f'{folder}; {id}; \r'
            csv_annotation.write(bin_img)
            id += 1
            
            im.close()

csv_annotation.close()

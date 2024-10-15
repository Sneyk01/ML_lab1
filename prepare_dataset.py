from PIL import Image
from numpy import asarray

need_rotate = True
img_count = 18
white = (255, 255, 255)
angels = [0, 5, 10, 15, 20, -5, -10, -15, -20]
folders = ["Photoshop", "Canva", "Corel", "Dreamweaver", "ExperienceDesign", "Flash", "Illustrator", "InDesign",
           "Pixlr", "Vectr"]

root_path = 'dataSet/Learning'

# rotate part
if need_rotate:
    for folder in folders:
        for img_index in range(img_count):
            img_path = f'{root_path}/{folder}/i{img_index}_0.png'
            try:
                im = Image.open(img_path)

                for angel in angels:
                    im_rotate = im.rotate(angel, fillcolor=white)
                    im_rotate.save(f'{root_path}/{folder}/i{img_index}_{angel}.png', quality=100)

                im.close()
            except:
                print(f'Can`t open: {img_path}')

# .csv part
id = 0
csv_annotation = open(f'{root_path}/annotation.csv', 'w')
for folder in folders:
    for img_index in range(img_count):
        for angel in angels:
            img_path = f'{root_path}/{folder}/i{img_index}_{angel}.png'
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

                bin_img += f'{folder}; {id}; \r'
                csv_annotation.write(bin_img)
                id += 1

                im.close()

            except:
                print(f'Can`t open: {img_path}')

csv_annotation.close()

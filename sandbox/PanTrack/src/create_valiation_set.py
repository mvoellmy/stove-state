import os
import random

path_train = '/home/miro/Desktop/M/train/'
path_val = '/home/miro/Desktop/M/val/'
img_type = '.jpg'
val_perc = 0.2

# get classes
class_list = [f for f in os.listdir(path_train) if os.path.isdir(os.path.join(path_train, f))]

print(class_list)
train_counter = 0
val_counter = 0

for label_nr, label_name in enumerate(class_list):
    img_list = [f for f in os.listdir(path_train + label_name) if os.path.isfile(os.path.join(path_train + label_name, f))
                and img_type in f]

    # check if validation images are there:
    val_img_list = [f for f in os.listdir(path_val + label_name) if os.path.isfile(os.path.join(path_val + label_name, f))
                and img_type in f]
    if val_img_list:
        input('Carefull!!! There are already validation images!')

    val_img_list = random.sample(img_list, int(len(img_list)*val_perc))
    train_counter += (len(img_list) - len(val_img_list))

    for img in val_img_list:
        os.rename(path_train + label_name + '/' + img, path_val + label_name + '/' + img)
        val_counter += 1

print('Train Set Size: {}'.format(train_counter))
print('Validation Set Size: {}'.format(val_counter))

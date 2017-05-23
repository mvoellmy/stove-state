import os
import random
import cv2
from helpers import mse

path_data = '/Users/miro/Desktop/test_dataset/'
path_data = '/Users/miro/Desktop/I_4/'
path_train = path_data + 'train/'
path_val = path_data + 'val/'
img_type = '.jpg'
delete_type = 'FULL'  # 'FULL' deletes img, 'MOVE' moves them to del folder


print('DATASET APPLICATION')
print('Parameters: \n\tDataset Path: {}\n\tImg Type: {}'.format(path_data, img_type))

print('1. Delete similar images')
print('2. Split dataset into test and train')
print('3. Undo split')
print("4. Only keep every n-th frame (for path-data)")
action = int(input('What do you want to do?\n'))

if action == 1:

    # Initialize Variables
    removed_counter = 0
    class_list = [f for f in os.listdir(path_train) if os.path.isdir(os.path.join(path_train, f))]

    # Check input
    threshold = float(input('Threshold for SSD [M=5, I=15] = '))

    for label_nr, label_name in enumerate(class_list):
        class_removed_counter = 0

        img_list = [f for f in os.listdir(path_train + label_name)
                    if os.path.isfile(os.path.join(path_train + label_name, f))
                    and img_type in f]

        for img_nr, img_name in enumerate(img_list):
            # Load img
            img = cv2.imread(path_train+label_name+'/'+img_name, 0)
            if img_nr > 0 and mse(img, old_img) < threshold:
                if delete_type == 'FULL':
                    os.remove(path_train + label_name + '/' + img_name)
                elif delete_type == 'MOVE':
                    os.rename(path_train + label_name + '/' + img_name, path_data + 'del/' + img_name)
                else:
                    print('ERROR: specify delete type!')

                class_removed_counter += 1
                removed_counter += 1

            old_img = img

        print('{} images deleted from {}'.format(class_removed_counter, label_name))

    print('_______________________________')
    print('{} images deleted in total'.format(removed_counter))

elif action == 2:

    # Initialize Variables
    train_counter = 0
    val_counter = 0
    # get classes
    class_list = [f for f in os.listdir(path_train) if os.path.isdir(os.path.join(path_train, f))]

    # Check input
    val_perc = float(input('Test % [0 bis 1] = '))  # Between 0 and 1
    for label_nr, label_name in enumerate(class_list):
        img_list = [f for f in os.listdir(path_train + label_name) if os.path.isfile(os.path.join(path_train + label_name, f))
                    and img_type in f]

        # check if validation images are there:
        val_img_list = [f for f in os.listdir(path_val + label_name) if os.path.isfile(os.path.join(path_val + label_name, f))
                        and img_type in f]
        if val_img_list:
            input('Carefull! There are already {0:.2f}% validation images! Run undo split first!'.
                  format(len(val_img_list)*100/(len(val_img_list)+len(img_list))))
            break

        val_img_list = random.sample(img_list, int(len(img_list)*val_perc))
        train_counter += (len(img_list) - len(val_img_list))

        for img_name in val_img_list:
            os.rename(path_train + label_name + '/' + img_name, path_val + label_name + '/' + img_name)
            val_counter += 1

    print('Train Set Size: {}'.format(train_counter))
    print('Validation Set Size: {}'.format(val_counter))

elif action == 3:

    # Initialize Variables
    moved_counter = 0
    # get classes
    class_list = [f for f in os.listdir(path_val) if os.path.isdir(os.path.join(path_val, f))]

    for label_nr, label_name in enumerate(class_list):
        # Get images in
        img_list = [f for f in os.listdir(path_val + label_name) if os.path.isfile(os.path.join(path_val + label_name, f))
                    and img_type in f]

        for img_name in img_list:
            os.rename(path_val + label_name + '/' + img_name, path_train + label_name + '/' + img_name)

            moved_counter += 1

    print('{} images were moved from val to train.'.format(moved_counter))

elif action == 4:
    # Initialize Variables
    removed_counter = 0
    class_list = [f for f in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, f))]

    # Check input
    frequency = float(input('Keep every n-th image.\nn = '))

    for label_nr, label_name in enumerate(class_list):
        class_removed_counter = 0

        img_list = [f for f in os.listdir(path_data + label_name)
                    if os.path.isfile(os.path.join(path_data + label_name, f))
                    and img_type in f]

        for img_nr, img_name in enumerate(img_list):
            # Load img
            img = cv2.imread(path_data+label_name+'/'+img_name, 0)
            if img_nr % frequency != 0:
                if delete_type == 'FULL':
                    os.remove(path_data + label_name + '/' + img_name)
                elif delete_type == 'MOVE':
                    os.rename(path_data + label_name + '/' + img_name, path_data + 'del/' + img_name)
                else:
                    print('ERROR: specify delete type!')

                class_removed_counter += 1
                removed_counter += 1

            old_img = img

        print('{} images deleted from {}'.format(class_removed_counter, label_name))

    print('_______________________________')
    print('{} images deleted in total'.format(removed_counter))


else:
    print('nothing done...wrong input')
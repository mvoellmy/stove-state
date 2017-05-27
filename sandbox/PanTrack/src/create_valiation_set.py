import os
import random
import cv2
import pickle
from helpers import *
import numpy as np

path_data = '/Users/miro/Desktop/test_dataset/'
path_data = '/Users/miro/Desktop/I_4/'  # 4.
path_data = '/Users/miro/Polybox/Shared/stove-state-data/ssds/pan_detect/data/'  # 5.
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
print("5. Find model accurracy")
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

elif action == 5:

    # Load model
    # pan_model_name = input('Model name = ')
    pan_model_name = '2017-05-18-18_25_11'  # I_4 begg
    pan_model_name = '2017-05-18-17_03_24'
    pan_models_path = '/Users/miro/Polybox/Shared/stove-state-data/ssds/pan_detect/pan_models/'

    # Load pan_model info file
    with open(pan_models_path + 'I_' + pan_model_name + '.txt', 'r') as file:
        _pan_params = eval(file.read())

    pan_model = pickle.load(open(pan_models_path + 'M_' + pan_model_name + '.sav', 'rb'))
    plate_of_interest = int(_pan_params['plate_of_interest'])
    corners = np.reshape(_pan_params['corners'], (-1, 4))

    # Init Accurracy matrix
    model_accurracy = np.zeros((3, 3))

    path_data = path_data + _pan_params['stove_type'] + '_' + str(_pan_params['plate_of_interest']) + '/pan/'
    # Get data
    class_list = [f for f in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, f))]

    label_comp_str = 'bla'
    old_label_comp_str = 'none'
    count = 0

    for label_nr, label_name in enumerate(class_list):

        img_list = [f for f in os.listdir(path_data + label_name)
                    if os.path.isfile(os.path.join(path_data + label_name, f))
                    and img_type in f]

        for img_nr, img_name in enumerate(img_list):
            # Load img
            img = cv2.imread(path_data+label_name+'/'+img_name, 0)

            patch = np.copy(img[corners[plate_of_interest - 1, 1]:corners[plate_of_interest - 1, 3],
                                corners[plate_of_interest - 1, 0]:corners[plate_of_interest - 1, 2]])

            patch_normalized = histogram_equalization(patch)

            pan_feature = get_HOG(patch_normalized,
                                  orientations=_pan_params['feature_params']['orientations'],
                                  pixels_per_cell=_pan_params['feature_params']['pixels_per_cell'],
                                  cells_per_block=_pan_params['feature_params']['cells_per_block'],
                                  widthPadding=_pan_params['feature_params']['widthPadding'])

            pan_label_predicted_id = pan_model.predict(pan_feature)
            pan_label_predicted_name = _pan_params['labels'][int(pan_label_predicted_id)]

            label_comp_str = '{} \t{}'.format(label_name, pan_label_predicted_name)

            count += 1
            if label_comp_str != old_label_comp_str:
                cv2.imshow(label_comp_str, patch)
                cv2.waitKey(1)
                print('...occured {1} times. \n {0} ...'.format(label_comp_str, count))
                count = 0

            old_label_comp_str = label_comp_str

            model_accurracy[label_nr, pan_label_predicted_id] += 1

        # Normalize Accurricies
        row_sum = model_accurracy.sum(axis=1)
        model_accurracy[label_nr, :] = model_accurracy[label_nr, :]/row_sum[label_nr]

    print('_______________________________')
    print('Accurracy for model: {}\n{}'.format(pan_model_name, model_accurracy))

    if input('Save accurracy? [y/n]\n') == 'y':
        accurracy_name = pan_models_path + 'A_' + pan_model_name + '.npy'
        np.save(accurracy_name, model_accurracy)

else:
    print('nothing done...wrong input')
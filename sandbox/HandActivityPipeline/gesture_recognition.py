from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from utils import *

np.set_printoptions(precision=2)

# [place pan, pour water, place egg, place lid, remove lid, remove egg, remove pan]
# becomes
# [pan, nothing, food, lid, lid, food, pan]
label_names = ['place pan', 'pour water', 'place egg', 'place lid', 'remove lid', 'remove egg', 'remove pan']
begg_labels = [1, -1, 2, 3, 4, 5, 6]


label_names = ['place', 'remove', 'season', 'other']
begg_labels = [1, -1, 1, 1, 2, 2, 2, 4]
segg_labels = [1, 4, 1, 1, 2, 3, 2, 2, 4]

label_names = ['pan-gesture', 'food-gesture', 'lid-gesture', 'season', 'stirr', 'other']
begg_labels = [1, -1, 2, 3, 3, 2, 1, 6]
segg_labels = [1, 6, 2, 3, 3, 4, 2, 1, 6]

label_names = ['place pan', 'place food', 'place lid', 'remove lid', 'remove food', 'remove pan', 'season', 'other']
begg_labels = [1, -1, 2, 3, 4, 5, 6, 8]
segg_labels = [1, 8, 2, 3, 4, 7, 5, 6, 8]
multiple_labels = [1, 2, 3, 4, 5, 6, 7, 8]

scegg_labels = [1, 6, 2, 2, 5, 4, 1]
file_names_begg = np.array(['I_2017-04-06-20_08_45_begg',
                            'I_2017-04-13-21_26_55_begg',
                            'I_20170419_232724_begg',       # -> 2 other gesture
                            'I_raspivid_20170421_begg',
                            'I_20170424_210116_begg',
                            'I_20170428_224946_begg',
                            'I_20170430_210819_begg',
                            'I_20170503_232838_begg',
                            'I_20170504_215844_begg',       # -> 1 other gesture
                            'I_20170505_214143_begg'])      # -> different position on stove

file_names_segg = np.array(['I_20170501_212055_segg',
                            'I_20170502_212256_segg',       # -> 2 other gestures
                            'I_20170503_234946_segg',       # -> 2 other gestures
                            'I_20170504_221703_segg',       # -> 3 other gestures
                            'I_20170505_220258_segg'])      # -> different position on stove

file_names_multiple = np.array(['I_20170516_212934_multiple0',
                                'I_20170516_212934_multiple1',
                                'I_20170516_212934_multiple2',
                                'I_20170516_212934_multiple3',
                                'I_20170516_212934_multiple4',
                                'I_20170516_214934_multiple0',
                                'I_20170516_214934_multiple1',
                                'I_20170516_214934_multiple2',
                                'I_20170516_214934_multiple3',
                                'I_20170516_214934_multiple4'])

count_correct_predictions = 0
count_predictions = 0
count_distinctive_predictions = 0
count_all_predictions = 0

for begg_val_idx in range(0,1):
    for segg_val_idx in range(0,1):
        for multiple_val_idx in range(0,1):
            path_features = 'gesture_features/begg/'
            out = extract_features(path_features, num_gestures=6, label_encoder=begg_labels, test_file_idx=begg_val_idx, file_names=file_names_begg)
            data_train, labels_train, labels_range_train, data_test, labels_test, labels_range_test = out

            # path_features = 'gesture_features/segg/'
            # out2 = extract_features(path_features, num_gestures=8, label_encoder=segg_labels, test_file_idx=segg_val_idx, file_names=file_names_segg)
            # data_train2, labels_train2, labels_range_train2, data_test2, labels_test2, labels_range_test2 = out2
            #
            # data_train = np.concatenate((data_train, data_train2))
            # labels_train = np.concatenate((labels_train, labels_train2))
            # labels_range_train = np.concatenate((labels_range_train, labels_range_train2[1:] + labels_range_train[-1]))
            # data_test = np.concatenate((data_test, data_test2))
            # labels_test = np.concatenate((labels_test, labels_test2))
            # labels_range_test = np.concatenate((labels_range_test, labels_range_test2[1:] + labels_range_test[-1]))

            path_features = 'gesture_features/multiple/'
            out3 = extract_features(path_features, num_gestures=8, label_encoder=multiple_labels, test_file_idx=multiple_val_idx, file_names=file_names_multiple)
            data_train3, labels_train3, labels_range_train3, data_test3, labels_test3, labels_range_test3 = out3

            data_train = np.concatenate((data_train, data_train3))
            labels_train = np.concatenate((labels_train, labels_train3))
            labels_range_train = np.concatenate((labels_range_train, labels_range_train3[1:] + labels_range_train[-1]))
            data_test = np.concatenate((data_test, data_test3))
            labels_test = np.concatenate((labels_test, labels_test3))
            labels_range_test = np.concatenate((labels_range_test, labels_range_test3[1:] + labels_range_test[-1]))

            fig = plt.figure()
            for i in range(0,len(labels_range_train)-1):
                plt.subplot(2,4,labels_train[i])
                a = labels_range_train[i]
                b = labels_range_train[i+1]
                plt.plot(data_train[a:b,1], -data_train[a:b,0])
                plt.title(label_names[labels_train[i]-1])
            fig.canvas.set_window_title('Trajectories of Training Data')


            N = 12 # Number of Spatio Temporal Features
            STF_train = spatio_temporal_features(data_train, labels_range_train, labels_train, N, label_names, 'Keyframes of Training Data')
            STF_test = spatio_temporal_features(data_test, labels_range_test, labels_test, N, label_names)

            def plot_STF(STF, labels, idx_feature, plot_name):
                N = STF.shape[2]
                num_labels = STF_train.shape[0]
                fig, ax = plt.subplots()
                cmap = plt.cm.get_cmap('jet',N)
                color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
                for i in range(0,num_labels):
                    plt.subplot(2,4,labels[i])
                    plt.scatter(range(0,N), STF[i,idx_feature,:], c=color[labels[i]-1])
                    plt.xlabel('Keyframes', fontsize=10)
                    plt.ylabel('Direction', fontsize=10)
                    plt.title(label_names[labels[i]-1])
                    plt.tight_layout()
                fig.canvas.set_window_title(plot_name)

            plot_STF(STF_train, labels_train, 0, 'STF - Trajectory orientation')
            plot_STF(STF_train, labels_train, 1, 'STF - Hand orientation')

            # Train each individual keyframe separately
            predicted = []
            counter = labels_test*0
            # print("Predict each Keyframe")
            # print("Correct predictions \t Predicted Labels")
            predictions = np.zeros((len(label_names), len(labels_test)), dtype=np.int)
            idx_gesture_to_plot = 4
            gesture_to_plot = np.zeros((len(label_names), N))
            clfs = []
            for i in range(0,N):
                # clf = GridSearchCV(SVC(), parameters, cv=2, n_jobs=1)
                clfs.append(svm.SVC())
                # clf = RandomForestClassifier(n_estimators=100)
                # clf.fit(STF_train[:,i].reshape(-1,1), labels_train)
                a = STF_train[:,:,i]
                clfs[i].fit(STF_train[:,:,i], labels_train)
                # predicted_labels = clf.predict(STF_test[:,i].reshape(-1,1))
                predicted_labels = clfs[i].predict(STF_test[:,:,i])
                predictions[np.array(predicted_labels)-1, np.array(range(0,len(labels_test)))] += 1
                gesture_to_plot[:,i] =  predictions[:,idx_gesture_to_plot]
                counter += (labels_test == predicted_labels) * 1
                # print("{} \t\t\t {}".format((labels_test == predicted_labels) * 1, predicted_labels))

            plt.figure()
            for i in range(0,len(label_names)):
                plt.plot(range(0,N), gesture_to_plot[i,:])

            plt.legend(label_names)

            print("Percentage of correct guesses")
            print(predictions[np.array(labels_test)-1, np.array(range(0,len(labels_test)))] / N)

            print("Accumulated Labels")
            print(predictions)
            sorted_labels = (-predictions).argsort(axis=0)
            predictions_1st = predictions[sorted_labels[0,:], np.array(range(0,len(labels_test)))] / N
            predictions_2nd = predictions[sorted_labels[1,:], np.array(range(0,len(labels_test)))] / N
            threshold = (predictions_1st - predictions_2nd) > 0.1
            print(threshold)
            final_predictions = sorted_labels[0,:] + 1


            final_predictions = final_predictions[threshold]
            labels_test = labels_test[threshold]
            count_distinctive_predictions += len(final_predictions)
            num_all_predictions = len(threshold)
            count_all_predictions += num_all_predictions

            num_correct_predictions = ((final_predictions == labels_test)*1).sum()
            num_predictions = len(labels_test)


            print("Final Predictions")
            print(final_predictions)
            print((final_predictions == labels_test)*1)
            print("Actual Labels")
            print(labels_test)
            print("Accuracy")
            print("{:.2f}% of {:.2f}% samples".format(num_correct_predictions/num_predictions*100, len(final_predictions)/num_all_predictions*100))
            print("begg: {}, multiple: {}".format(begg_val_idx, multiple_val_idx))

            count_correct_predictions += num_correct_predictions
            count_predictions += num_predictions

print("Total Accuracy")
print("{:.2f}% of {:.2f}% samples".format(count_correct_predictions/count_predictions*100, count_distinctive_predictions/count_all_predictions*100))

# print(predictions.transpose())

# plt.show()

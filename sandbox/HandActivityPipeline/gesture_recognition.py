from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from utils import *



# [place pan, pour water, place egg, place lid, remove lid, remove egg, remove pan]
# becomes
# [pan, nothing, food, lid, lid, food, pan]
label_names = ['place pan', 'pour water', 'place egg', 'place lid', 'remove lid', 'remove egg', 'remove pan']
begg_labels = [1, -1, 2, 3, 4, 5, 6]


label_names = ['place', 'remove', 'season', 'other']
begg_labels = [1, -1, 1, 1, 2, 2, 2, 4]
segg_labels = [1, 1, 1, 1, 2, 3, 2, 2, 4]

# label_names = ['pan-gesture', 'food-gesture', 'lid-gesture', 'season', 'stirr', 'other']
# begg_labels = [1, -1, 2, 3, 3, 2, 1]
# segg_labels = [1, 6, 2, 3, 3, 4, 2, 1]

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

count_correct_predictions = 0
count_predictions = 0
for begg_val_idx in range(0,1):
    for segg_val_idx in range(0,1):
        path_features = 'gesture_features/begg/'
        out = extract_features(path_features, num_gestures=6, label_encoder=begg_labels, test_file_idx=begg_val_idx, file_names=file_names_begg)
        data_train, labels_train, labels_range_train, data_test, labels_test, labels_range_test = out

        path_features = 'gesture_features/segg/'
        out2 = extract_features(path_features, num_gestures=8, label_encoder=segg_labels, test_file_idx=segg_val_idx, file_names=file_names_segg)
        data_train2, labels_train2, labels_range_train2, data_test2, labels_test2, labels_range_test2 = out2

        data_train = np.concatenate((data_train, data_train2))
        labels_train = np.concatenate((labels_train, labels_train2))
        labels_range_train = np.concatenate((labels_range_train, labels_range_train2[1:] + labels_range_train[-1]))
        data_test = np.concatenate((data_test, data_test2))
        labels_test = np.concatenate((labels_test, labels_test2))
        labels_range_test = np.concatenate((labels_range_test, labels_range_test2[1:] + labels_range_test[-1]))

        fig = plt.figure()
        for i in range(0,len(labels_range_train)-1):
            plt.subplot(2,3,labels_train[i])
            a = labels_range_train[i]
            b = labels_range_train[i+1]
            plt.plot(data_train[a:b,1], -data_train[a:b,0])
            plt.title(label_names[labels_train[i]-1])
        fig.canvas.set_window_title('Trajectories of Training Data')


        N = 10 # Number of Spatio Temporal Features
        STF_train = spatio_temporal_features(data_train, labels_range_train, labels_train, N, label_names, 'Keyframes of Training Data')
        STF_test = spatio_temporal_features(data_test, labels_range_test, labels_test, N, label_names)


        fig, ax = plt.subplots()
        cmap = plt.cm.get_cmap('jet',N)
        color = ['b', 'r', 'y', 'g', 'm', 'c']
        for i in range(0,len(labels_range_train)-1):
            plt.subplot(2,3,labels_train[i])
            plt.scatter(range(0,N), STF_train[i,0,:], c=color[labels_train[i]-1])
            plt.xlabel('Keyframes', fontsize=10)
            plt.ylabel('Direction', fontsize=10)
            plt.title(label_names[labels_train[i]-1])
            plt.tight_layout()
        fig.canvas.set_window_title('Spatio Temporal Features (STF)')

        print("Actual Labels")
        print(labels_test)

        # Train using all keyframes as a feature vector
        # parameters = [
        #         # {'kernel': ['rbf'], 'gamma': [1, 0.1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
        #         #{'kernel': ['linear'], 'C': [10, 100, 1000]},
        #         {'kernel': ['poly'], 'gamma': [0.2], 'C': [100]}
        #     ]
        # clf = GridSearchCV(SVC(), parameters, cv=2, n_jobs=1)
        # clf.fit(STF_train[:,0,:], labels_train)
        # predicted_labels = clf.predict(STF_test[:,0,:])
        # print("Correct predictions \t Predicted Labels")
        # print("{} \t\t\t {}".format((labels_test==predicted_labels)*1, predicted_labels))

        # Train each individual keyframe separately
        predicted = []
        counter = labels_test*0
        print("Predict each Keyframe")
        print("Correct predictions \t Predicted Labels")
        predictions = np.zeros((len(label_names), len(labels_test)))
        for i in range(0,N):
            # clf = GridSearchCV(SVC(), parameters, cv=2, n_jobs=1)
            clf = svm.SVC()
            # clf = RandomForestClassifier(n_estimators=100)
            # clf.fit(STF_train[:,i].reshape(-1,1), labels_train)
            clf.fit(STF_train[:,:,i], labels_train)
            # predicted_labels = clf.predict(STF_test[:,i].reshape(-1,1))
            predicted_labels = clf.predict(STF_test[:,:,i])
            predictions[np.array(predicted_labels)-1, np.array(range(0,len(labels_test)))] += 1
            counter += (labels_test == predicted_labels) * 1
            print("{} \t\t\t {}".format((labels_test == predicted_labels) * 1, predicted_labels))

        print("Percentage of correct guesses")
        print(counter/N)
        print((counter/N >= 0.5)*1)
        count_correct_predictions += ((counter/N >= 0.5)*1).sum()
        count_predictions += len(counter)

print("Accuracy")
print(count_correct_predictions/count_predictions)

print(predictions)

plt.show()

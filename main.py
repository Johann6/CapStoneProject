import machinery
from machinery import coordinates
from machinery import targetDim
from tqdm import tqdm
import pickle
import numpy as np
#from sklearn.datasets import load_files       

              
import matplotlib.pyplot as plt 
#from glob import glob
#from keras.preprocessing import image
#from keras.utils import np_utils

import keras_vggface.utils
import keras.applications.imagenet_utils
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from pathlib import Path
import types
import tempfile
import keras.models
import os



####### Global Var's #######




if __name__ == "__main__":

    

    (face_jpgs_paths, face_tensors, face_targets) = machinery.load_dataset()

    #normalize tensors & targets
    # TODO: divide by 255? check max values!
    #face_tensors = paths_to_tensor(face_jpgs_paths).astype('float32')/255
    face_tensors = keras_vggface.utils.preprocess_input(face_tensors, version=1)   #FaceNet VGG16
    #face_tensors = keras.applications.imagenet_utils.preprocess_input(paths_to_tensor(face_jpgs_paths))   #ImageNet

    face_targets = np.array(face_targets) / targetDim.x

    # TODO: remove!
    # limit dataset size!
    face_jpgs_paths = face_jpgs_paths[:1000]
    face_tensors = face_tensors[:1000]
    face_targets = face_targets[:1000]

    # check if normalization was messed up...
    #if (np.max([np.max(tensor) for tensor in face_tensors]) > 1.0) or (np.min([np.max(tensor) for tensor in face_tensors]) < 0.0):
    #    assert(0)
    if (np.max([np.max(target) for target in face_targets]) > 1.0) or (np.min([np.max(target) for target in face_targets]) < 0.0):
        assert(0)

    if Path(os.path.dirname(os.path.realpath(__file__)) + "\\bottleneck_features.p").is_file():
        print('Load bottleneck features from pickle.')
        bottleneck_features = pickle.load( open( "bottleneck_features.p", "rb" ) )
    else:
        print('Load Base model.')
        base_model = machinery.getBaseModel()

        print('Extract bottleneck features.')
        list_of_features = [base_model.predict(np.expand_dims(tensor, axis=0)) for tensor in tqdm(face_tensors)]
        bottleneck_features = np.vstack(list_of_features)
        # Save tensors into a pickle file.
        print('Dump bottleneck features.')
        pickle.dump( bottleneck_features, open( "bottleneck_features.p", "wb" ) )



    # We got our tensors, targets and bottleneck_features ready at this stage!

    
    X = bottleneck_features
    y = face_targets

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)

    print("Get model.")
    InceptionV3_model = machinery.getModel(bottleneck_features.shape[1:])

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.InceptionV3.hdf5', 
                               verbose=1, save_best_only=True)
    early_Stopping = EarlyStopping(monitor='val_loss', min_delta=1, patience=3, verbose=1, mode='auto')

    trainModel_b = True
    if trainModel_b:
        print('Train regression model.')
        #InceptionV3_model.load_weights('saved_models/weights.best.InceptionV3.hdf5')
        InceptionV3_model.fit(X_train, y_train, 
                  validation_data=(X_valid, y_valid),
                  epochs=20, batch_size=20, callbacks=[checkpointer, early_Stopping], verbose=1)
#                  epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

    InceptionV3_model.load_weights('saved_models/weights.best.InceptionV3.hdf5')

    print('Test complete model.')
    predictions = []

    #predict on face_tensors base_model.predict(np.expand_dims(face_tensors[index], axis=0)) == bottleneck_features[index]
    for index in range(10):
        result = InceptionV3_model.predict(np.expand_dims(bottleneck_features[index], axis=0))
        #result = base_model.predict(np.expand_dims(face_tensors[index], axis=0))
        predictions.append(result[0])

    predictions = np.array(predictions) * targetDim.x

    for index in range(10):
        print(
        ' delta x: ' + str(predictions[index][0] - face_targets[index][0]) +
        ' delta y: ' + str(predictions[index][1] - face_targets[index][1]) +
        ' delta w: ' + str(predictions[index][2] - face_targets[index][2]) +
        ' delta h: ' + str(predictions[index][3] - face_targets[index][3])
        )
    machinery.showImg(face_jpgs_paths[:10], predictions, targetDim)
    #showImg(face_jpgs_paths[10:20], face_targets[10:20], targetDim)
    #showImg(face_jpgs_paths[20:30], face_targets[20:30], targetDim)

    plt.show()






    print('Total number of relevant pictures: ' + str(len(face_tensors)))

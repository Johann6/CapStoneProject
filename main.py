import machinery
from machinery import coordinates
from machinery import targetDim
#from sklearn.datasets import load_files       

              
import matplotlib.pyplot as plt 
#from glob import glob
#from keras.preprocessing import image
#from keras.utils import np_utils


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from pathlib import Path
import types
import tempfile
import keras.models
import os


####### Global Var's #######



def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__






if __name__ == "__main__":

    (face_jpgs_paths, face_tensors, face_targets) = load_dataset()

    #img_path = 'elephant.jpg'
    #img = image.load_img(img_path, target_size=(224, 224))
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)

    if Path(os.path.dirname(os.path.realpath(__file__)) + "\\bottleneck_features.p").is_file():
        print('Load bottleneck features from pickle.')
        bottleneck_features = pickle.load( open( "bottleneck_features.p", "rb" ) )
    else:
        print('Load InceptionV3 model.')
        base_model = InceptionV3(weights='imagenet', include_top=False)
        #make_keras_picklable()
        ## Save InceptionV3_model into a pickle file.
        #pickle.dump( base_model, open( "InceptionV3_model.p", "wb" ) )

        print('Extract bottleneck features.')
        list_of_features = [base_model.predict(np.expand_dims(tensor, axis=0)) for tensor in tqdm(face_tensors)]
        bottleneck_features = np.vstack(list_of_features)
        # Save tensors into a pickle file.
        print('Dump bottleneck features.')
        pickle.dump( bottleneck_features, open( "bottleneck_features.p", "wb" ) )

    X = bottleneck_features
    y = face_targets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)



    InceptionV3_model = Sequential()
    InceptionV3_model.add(GlobalAveragePooling2D(input_shape=bottleneck_features.shape[1:]))
    InceptionV3_model.add(Dense(5000, activation='relu'))
    InceptionV3_model.add(Dense(4))
    InceptionV3_model.summary()


    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    InceptionV3_model.compile(loss='mean_squared_error', optimizer=adam)

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.InceptionV3.hdf5', 
                               verbose=1, save_best_only=True)

    early_Stopping = EarlyStopping(monitor='val_loss', min_delta=1, patience=3, verbose=1, mode='auto')

    trainModel_b = False
    if trainModel_b:
        print('Train regression model.')
        InceptionV3_model.load_weights('saved_models/weights.best.InceptionV3.hdf5')
        InceptionV3_model.fit(X_train, y_train, 
                  validation_data=(X_valid, y_valid),
                  epochs=20, batch_size=20, callbacks=[checkpointer, early_Stopping], verbose=1)

    InceptionV3_model.load_weights('saved_models/weights.best.InceptionV3.hdf5')

    print('Test complete model.')
    predictions = []

    #predict on face_tensors base_model.predict(np.expand_dims(face_tensors[index], axis=0)) == bottleneck_features[index]
    for index in range(10):
        result = InceptionV3_model.predict(np.expand_dims(bottleneck_features[index], axis=0))
        #result = base_model.predict(np.expand_dims(face_tensors[index], axis=0))
        predictions.append(result[0])

    for index in range(10):
        print(
        ' delta x: ' + str(predictions[index][0] - face_targets[index][0]) +
        ' delta y: ' + str(predictions[index][1] - face_targets[index][1]) +
        ' delta w: ' + str(predictions[index][2] - face_targets[index][2]) +
        ' delta h: ' + str(predictions[index][3] - face_targets[index][3])
        )
    showImg(face_jpgs_paths[:10], predictions, targetDim)
    #showImg(face_jpgs_paths[10:20], face_targets[10:20], targetDim)
    #showImg(face_jpgs_paths[20:30], face_targets[20:30], targetDim)

    plt.show()






    print('Total number of relevant pictures: ' + str(len(face_tensors)))

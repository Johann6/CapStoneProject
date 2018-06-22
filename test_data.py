import machinery
from machinery import coordinates
from machinery import targetDim
from os import listdir
from os.path import isfile, join, dirname, realpath
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt


from keras_vggface.vggface import VGGFace
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D


############## Global Var's ##############
working_dir = "C:/Users/PC/Documents/ML - Udacity_MLND/Capstone_Project_face_localization/Test_dataset/"


if __name__ == "__main__":
    files = [f for f in listdir(working_dir) if isfile(join(working_dir, f)) & f.endswith('.jpg')]

    face_jpgs_paths = [working_dir+f for f in files]

    face_tensors = machinery.paths_to_tensor(face_jpgs_paths).astype('float32')/255

    #face_tensors = [np.expand_dims(face_tensor.transpose((2,0,1)), axis=0) for face_tensor in face_tensors]
    #face_tensors = np.vstack(face_tensors)

    if Path(working_dir+"bottleneck_features_testImages.p").is_file():
        print('Load bottleneck features from pickle.')
        bottleneck_features = pickle.load( open( working_dir+"bottleneck_features_testImages.p", "rb" ) )
    else:
        print('Load InceptionV3 model.')
        #base_model = InceptionV3(weights='imagenet', include_top=False)
        #base_model = VGGFace(include_top=True, model='vgg16', input_shape=(224, 224, 3), pooling='avg')
        base_model = VGGFace(include_top=True, model='vgg16', pooling='avg')
        #make_keras_picklable()
        ## Save InceptionV3_model into a pickle file.
        #pickle.dump( base_model, open( "InceptionV3_model.p", "wb" ) )

        print('Extract bottleneck features.')
        list_of_features = [base_model.predict(np.expand_dims(tensor, axis=0)) for tensor in tqdm(face_tensors)]
        bottleneck_features = np.vstack(list_of_features)
        # Save tensors into a pickle file.
        print('Dump bottleneck features.')
        pickle.dump( bottleneck_features, open(working_dir+"bottleneck_features_testImages.p", "wb" ) )

    print("Create Top model based upon InceptionV3 model.")
    InceptionV3_model = machinery.getModel(bottleneck_features.shape[1:])

    InceptionV3_model.load_weights('saved_models/weights.best.InceptionV3.hdf5')

    predictions = []

    print("Make predictions.")
    #predict on face_tensors base_model.predict(np.expand_dims(face_tensors[index], axis=0)) == bottleneck_features[index]
    for feature in bottleneck_features:
        result = InceptionV3_model.predict(np.expand_dims(feature, axis=0))
        #result = base_model.predict(np.expand_dims(face_tensors[index], axis=0))
        predictions.append(result[0])

    predictions = np.array(predictions) * targetDim.x

    machinery.showImg(face_jpgs_paths, predictions, targetDim)

    plt.show()






    print('Total number of relevant pictures: ' + str(len(face_tensors)))

from tqdm import tqdm
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path



from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam, SGD

from keras_vggface.vggface import VGGFace
from keras.engine import  Model
from keras.layers import Input


class coordinates:
  x = 0.0
  y = 0.0

targetDim = coordinates()
targetDim.x = 224
targetDim.y = 224


def extractUnixPaths(list_of_tuple):
    count = 0
    analyseNextLine = False
    getNextLine = False

    unix_paths_to_jpgs = list()
    face_targets = list()

    for file_path_pair in list_of_tuple:
        file_obj = file_path_pair[0]
        root_dir = file_path_pair[1]
        for line in file_obj:
            if getNextLine == True:
                target = np.fromstring(line, dtype=int, sep=" ")
                target = target[:4]
                if np.array_equal(target, [0, 0, 0, 0]):
                    #discard, remove last entry in unix_paths_to_jpgs
                    del unix_paths_to_jpgs[-1]
                    print('Invalid target discarded! target=' + str(target))
                else:
                    face_targets.append(target)
                getNextLine = False
            if analyseNextLine == True:
                if line == '1\n':
                    count = count +1
                    #print(root_dir + path)
                    unix_paths_to_jpgs.append(root_dir + path[:len(path)-1])
                    getNextLine = True
                analyseNextLine = False

            if line.endswith('jpg\n'):
                path = line
                analyseNextLine = True
    return ( np.array(unix_paths_to_jpgs), np.array(face_targets) )

def unpackTuple(path_target_tuple):
    path_str = path_target_tuple[0]
    target = path_target_tuple[1]
    return (path_str, target)

def showImg(path_strs, targets, targetDim):
    list_of_tuple = list(zip(path_strs, targets))
    img_array = []

    for a_tuple in list_of_tuple:
        path_str = a_tuple[0]
        target = a_tuple[1]
        # load color (BGR) image
        img = cv2.imread(path_str)
        #resize
        img = cv2.resize(img, (targetDim.x, targetDim.y)) 
        # convert BGR image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _1st_point = (target[0], target[1])
        _2nd_point = (target[0]+target[2], target[1]+target[3])
        #print('_1st_point: ' + str(_1st_point))
        #print('_2nd_point: ' + str(_2nd_point))

        cv2.rectangle(img,_1st_point,_2nd_point,(0,0,255),2)
        img_array.append(img)

    f, axarr = plt.subplots(2, 5)
    axarr[0, 0].imshow(img_array[0])
    axarr[0, 1].imshow(img_array[1])
    axarr[0, 2].imshow(img_array[2])
    axarr[0, 3].imshow(img_array[3])
    axarr[0, 4].imshow(img_array[4])

    axarr[1, 0].imshow(img_array[5])
    axarr[1, 1].imshow(img_array[6])
    axarr[1, 2].imshow(img_array[7])
    axarr[1, 3].imshow(img_array[8])
    axarr[1, 4].imshow(img_array[9])

    #plt.show()

def path_to_tensor(face_jpgs_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(face_jpgs_path, target_size=(targetDim.x, targetDim.y))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def debug_func(path_target_tuple):
    return 1

def paths_to_tensor(face_jpgs_paths):
    print("Create Tensor Array from files.")
    #debug = [debug_func(path_target_tuple) for path_target_tuple in tqdm(path_target_tuples)]
    #[path_to_tensor(path_target_tuple) for path_target_tuple in tqdm(path_target_tuples)]
    list_of_tensors = [path_to_tensor(face_jpgs_path) for face_jpgs_path in tqdm(face_jpgs_paths)]
    return np.vstack(list_of_tensors)

def normTarget(path_target_tuple):
    (img_path_str, target) = unpackTuple(path_target_tuple)

    img_size = image.load_img(img_path_str).size
    img_source_dim = coordinates()
    img_source_dim.x = img_size[0]
    img_source_dim.y = img_size[1]
    img_target_dim = targetDim

    # modify bounding box
    target[0] = img_target_dim.x * target[0] / img_source_dim.x  #x
    target[1] = img_target_dim.y * target[1] / img_source_dim.y  #y
    target[2] = img_target_dim.x * target[2] / img_source_dim.x  #width (x)
    target[3] = img_target_dim.y * target[3] / img_source_dim.y  #height (y)

    return target

def normTargets(path_target_tuples):
    print("Normalize Targets.")
    list_of_targets = [normTarget(path_target_tuple) for path_target_tuple in tqdm(path_target_tuples)]
    return list_of_targets

# define function to load train, test, and validation datasets
def load_dataset_from_files():
    root_dir = 'C:/Users/PC/Documents/ML - Udacity_MLND/Capstone_Project_face_localization/WIDER_Face_dataset/'
    dir_val = 'WIDER_val/'
    filename_val = 'wider_face_val_bbx_gt.txt'
    dir_train = 'WIDER_train/'
    filename_train = 'wider_face_train_bbx_gt.txt'
    
    fobj_train = open(root_dir + dir_train + filename_train, "r")
    fobj_val = open(root_dir + dir_val + filename_val, "r")

    list_of_tuple = [(fobj_train, root_dir+dir_train), (fobj_val, root_dir+dir_val)]
       
    #path_target_tuples = extractUnixPaths(list_of_tuple)
    (face_jpgs_paths, face_targets) = extractUnixPaths(list_of_tuple)

    face_targets = normTargets(list(zip(face_jpgs_paths, face_targets)))

    #showImg(face_jpgs_paths[:10], face_targets[:10], targetDim)

    face_tensors = paths_to_tensor(face_jpgs_paths)

    for tuple in list_of_tuple:
        file = tuple[0]
        file.close()

    return (face_jpgs_paths, face_tensors, face_targets)

def load_dataset():
    working_dir = os.path.dirname(os.path.realpath(__file__))
    files = []
    files.append(Path(working_dir + "\\face_tensors.p"))
    files.append(Path(working_dir + "\\face_targets.p"))
    files.append(Path(working_dir + "\\face_jpgs_paths.p"))

    usePickle_b = True
    for file in files:
        if not file.is_file():
            usePickle_b = False

    if usePickle_b:
        print("Load dataset from pickles.")
        face_tensors = pickle.load( open( "face_tensors.p", "rb" ) )
        face_targets = pickle.load( open( "face_targets.p", "rb" ) )
        face_jpgs_paths = pickle.load( open( "face_jpgs_paths.p", "rb" ) )
        print(str(face_jpgs_paths.shape[0]) + " tensors loaded.")
    else:
        print("Load dataset from files and create pickles.")
        (face_jpgs_paths, face_tensors, face_targets) = load_dataset_from_files()
        # Save tensors into a pickle file.
        pickle.dump( face_tensors, open( "face_tensors.p", "wb" ) )
        pickle.dump( face_targets, open( "face_targets.p", "wb" ) )
        pickle.dump( face_jpgs_paths, open( "face_jpgs_paths.p", "wb" ) )

    #check if arrays plausible
    assert((face_jpgs_paths.shape[0] == face_tensors.shape[0]) & (face_jpgs_paths.shape[0] == len(face_targets)))

    return (face_jpgs_paths, face_tensors, face_targets)


def getModel(inputShape):
#Ideen: BatchNormalization? DropOut? Augmentation of target vec?

    #model = Sequential()
    #model.add(GlobalAveragePooling2D(input_shape=inputShape))
    #model.add(Dense(5000, activation='relu'))
    #model.add(Dense(4))
    #model.summary()

    #model = Sequential()
    #model.add(GlobalAveragePooling2D(input_shape=bottleneck_features.shape[1:]))
    #model.add(Dense(5000))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Dense(4))
    #model.summary()

    #model = Sequential()
    #model.add(GlobalAveragePooling2D(input_shape=inputShape))
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dense(2048, activation='relu'))
    #model.add(Dense(1024, activation='relu'))
    #model.add(Dense(4))
    #model.summary()

    #model = Sequential()
    ##model.add(GlobalAveragePooling2D(input_shape=inputShape))
    ##model.add(Dropout(0.3))
    #model.add(Dense(4096, input_shape=inputShape))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(2048))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(1024))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Dense(4))
    #model.summary()

    model = Sequential()
    model.add(BatchNormalization(input_shape=inputShape))
    model.add(MaxPooling2D())
    model.add(Flatten(name='flatten'))
    model.add(Dense(2048, name='fc6'))
    model.add(Activation('relu', name='fc6/relu'))
    model.add(Dense(1024, name='fc7'))
    model.add(Activation('relu', name='fc7/relu'))
    model.add(Dense(4, name='fc8'))
    model.summary()

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


def getBaseModel():
    #base_model = InceptionV3(weights='imagenet', include_top=False)
    #base_model = VGGFace(include_top=False, model='vgg16', input_shape=(224, 224, 3), pooling='max')
    #base_model = VGGFace(include_top=False, model='vgg16', pooling='avg')

    # Layer Features
    layer_name = 'pool4'
    vgg_model = VGGFace(input_shape=(224, 224, 3))
    out = vgg_model.get_layer(layer_name).output
    base_model = Model(vgg_model.input, out)

    return base_model

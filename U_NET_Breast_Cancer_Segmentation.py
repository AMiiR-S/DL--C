""" @author: amiir"""
'''****** U_NET: Deep Learning Model to Segment Breast Cancerous Cells******'''

# 1. Import Required Modules

import os
import glob
import keras
import random
import numpy as np
import tensorflow as tf
from keras.layers import *
import keras.backend as k
from keras.models import *
from keras.optimizers import *
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread, imshow, imsave
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping



# 2. Define Train & Test Path (Images + Masks Path for Train and Test Stages)

TRAIN_IMAGE_PATH = '/home/amiir/Downloads/DataSets/Medical/Breast Cancer Cell Segmentation2/Inputs_Train/'
TRAIN_MASK_PATH = '/home/amiir/Downloads/DataSets/Medical/Breast Cancer Cell Segmentation2/Masks_Train/'
TEST_IMAGE_PATH = '/home/amiir/Downloads/DataSets/Medical/Breast Cancer Cell Segmentation2/Inputs_Test/'
TEST_MASK_PATH = '/home/amiir/Downloads/DataSets/Medical/Breast Cancer Cell Segmentation2/Masks_Test/'

# 3. Initialize Images & Masks Size

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 256, 256, 3

# 4. Define Pre_Processing Function ( Region of Interest Extraction_ROI)

Train_Mask_List = sorted(next(os.walk(TRAIN_MASK_PATH))[2])
Test_Mask_List = sorted(next(os.walk(TEST_MASK_PATH))[2])


def Data_PreProcessing_Train():
    Init_Image = np.zeros((len(Train_Mask_List), 768, 896, 3), dtype = np.uint8)
    Init_Mask = np.zeros((len(Train_Mask_List), 768, 896), dtype = np.bool)
    Train_X = np.zeros((len(Train_Mask_List), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
    Train_Y = np.zeros((len(Train_Mask_List), IMG_HEIGHT, IMG_WIDTH,1), dtype = np.bool)
    
    n = 0
    
    for mask_path in glob.glob('{}/*.TIF'.format(TRAIN_MASK_PATH)):
        
        
        base = os.path.basename(mask_path)
        image_ID, ext = os.path.splitext(base)
        image_path = '{}/{}_ccd.tif'.format(TRAIN_IMAGE_PATH, image_ID)
        mask = imread(mask_path)
        image = imread(image_path)
        
        y_coord, x_coord = np.where(mask == 255)
        y_min = min(y_coord)
        y_max = max(y_coord)
        x_min = min(x_coord)
        x_max = max(x_coord)
        
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]
        
        Train_X[n] = resize(cropped_image[:, :, :IMG_CHANNELS], (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                            mode = 'constant',
                            anti_aliasing=True,
                            preserve_range=True)
        
        Train_Y[n] = np.expand_dims(resize(cropped_mask, (IMG_HEIGHT, IMG_WIDTH),
                            mode = 'constant',
                            anti_aliasing=True,
                            preserve_range=True), axis = -1)
        
        Init_Image[n] = image
        Init_Mask[n] = mask
        
        n+=1
        
    return Train_X, Train_Y, Init_Image, Init_Mask

Train_Inputs, Train_Masks, Init_Image, Init_Mask = Data_PreProcessing_Train()

        

def Data_PreProcessing_Test():
                            
    Test_X = np.zeros((len(Test_Mask_List), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
    Test_Y = np.zeros((len(Test_Mask_List), IMG_HEIGHT, IMG_WIDTH,1), dtype = np.bool)
    
    n = 0
    
    for mask_path in glob.glob('{}/*.TIF'.format(TEST_MASK_PATH)):
        
        
        base = os.path.basename(mask_path)
        image_ID, ext = os.path.splitext(base)
        image_path = '{}/{}_ccd.tif'.format(TEST_IMAGE_PATH, image_ID)
        mask = imread(mask_path)
        image = imread(image_path)
        
        y_coord, x_coord = np.where(mask == 255)
        y_min = min(y_coord)
        y_max = max(y_coord)
        x_min = min(x_coord)
        x_max = max(x_coord)
        
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]
        
        Test_X[n] = resize(cropped_image[:, :, :IMG_CHANNELS], (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                            mode = 'constant',
                            anti_aliasing=True,
                            preserve_range=True)
        
        Test_Y[n] = np.expand_dims(resize(cropped_mask, (IMG_HEIGHT, IMG_WIDTH),
                            mode = 'constant',
                            anti_aliasing=True,
                            preserve_range=True), axis = -1)
        
       
        n+=1
        
    return Test_X, Test_Y

Test_Inputs, Test_Masks = Data_PreProcessing_Test()
 
        
        
        

# 4.1. Show The Results for Preprocessing Stage

print('Original_Image')
imshow(Init_Image[0])
plt.show()


print('Original_Mask')
imshow(Init_Mask[0])
plt.show()

print('Region_of_Interested_Image')
imshow(Train_Inputs[0])
plt.show()

print('Region_of_Interested_Mask')
imshow(np.squeeze(Train_Masks[0]))
plt.show()


rows = 1
columns = 4
Figure = plt.figure(figsize=(15,15))
Image_List = [Init_Image[0], Init_Mask[0], Train_Inputs[0], Train_Masks[0]]
     
for i in range(1,rows*columns +1):
    Image = Image_List[i-1]
    Sub_Plot_Image = Figure.add_subplot(rows, columns, i)
    Sub_Plot_Image.imshow(np.squeeze(Image))
plt.show()
              
             

# 5. Implementain of U_NET model for Semantic Segmentation

def U_NET_Segmentation(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    
    inputs = Input(input_size)
    n = Lambda(lambda x:x/255)(inputs)
    
    
    
    
    c1 = Conv2D(16, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(n)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)

    c2 = Conv2D(32, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)
    
    c3 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)
   
    c4 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2,2))(c4)
   
    c5 = Conv2D(256, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)
    
    
    u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)
        
    u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)
        
    u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)
        
    u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)
       


    outputs = Conv2D(1,(1,1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[Mean_IOU_Evaluator])
    
    model.summary()
    
    return model


# 6. Define U_NET Model Evaluator (Intersection Over Union_IOU)

def Mean_IOU_Evaluator(y_true, y_pred):
    
    prec = []
    
    for t in np.arange(0.5, 1, 0.05):
        
        y_pred_ = tf.cast(y_pred>t, dtype = tf.int32)
        
        score, up_opt = tf.py_function(tf.keras.metrics.MeanIoU,[y_true, y_pred_, 2], Tout=tf.float64)
        k.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
            prec.append(score)
    return k.mean(k.stack(prec), axis= 0)

model = U_NET_Segmentation()


# 7. Show The Results per Epoch

class loss_history(keras.callbacks.Callback):
    
    def __init__(self, x=4):
        self.x = x
        
    def on_epoch_begin(self, epoch, logs={}):
        
        imshow(Train_Inputs[self.x])
        plt.show()
        
        imshow(np.squeeze(Train_Masks[self.x]))
        plt.show()
        
        preds_train = self.model.predict(np.expand_dims(Train_Inputs[self.x], axis = 0))
        imshow(np.squeeze(preds_train[0]))
        plt.show()


imageset= 'BCC'
backbone= 'UNET'
version= 'v1.0'
model_h5= 'model-{imageset}-{backbone}-{version}.h5'.format(imageset=imageset, backbone=backbone, version=version)
model_h5_checkpoint = '{model_h5}.checkpoint'.format(model_h5=model_h5)

        
earlystopper = EarlyStopping(patience=7, verbose=1)
checkpointer = ModelCheckpoint(model_h5_checkpoint, verbose=1, save_best_only=True)


# 8. Train U_NET Model using Training Samples

results = model.fit(Train_Inputs, Train_Masks, validation_split=0.1, batch_size=2, epochs=50,
                    callbacks=[earlystopper, checkpointer, loss_history()])


# 9. U_NET Model Evaluation using Test Samples

preds_train = model.predict(Train_Inputs, verbose=1)
preds_train_t = (preds_train>0.5).astype(np.uint8)
preds_test = model.predict(Test_Inputs, verbose=1)
preds_test_t = (preds_test>0.5).astype(np.uint8)

# 10. Show Final Results (Segmented Images)

ix = random.randint(0, len(Train_Inputs)-1)
print(ix)

print('Train_Image')
imshow(Train_Inputs[ix])
plt.show()

print('Train_Masks')
imshow(np.squeeze(Train_Masks[ix]))
plt.show()

print('Segment_Image')
imshow(np.squeeze(preds_train[ix]))
plt.show()


iix = random.randint(0, 1)
print(iix)

print('Test_Image')
imshow(Test_Inputs[iix])
plt.show()

print('Test_Masks')
imshow(np.squeeze(Test_Masks[iix]))
plt.show()

print('Segment_Test_Mask')
imshow(np.squeeze(preds_test[iix]))
plt.show()


































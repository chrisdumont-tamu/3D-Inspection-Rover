# import ML and Visualization libraries
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

#####################################################################
## Functions 
def showDsImage(dataset):
    # Show images in the dataset
    for image_batch, label_batch in dataset.take(1):
        plt.imshow(image_batch[0].numpy().astype("uint8"))
        plt.show()

def get_dataset_partitions_tf(ds, train_split=0.9, shuffle=True, shuffle_size=10000):
    # This function partitions the full dataset into subsets used for training, and validation
    # Validation partition is 1 - train_split
    
    # ds - dataset, train_split - % of dataset that is used for training, val_split - % of dataset that is used for validation
    # shuffle - if we want to shuffle the dataset when we take it, shuffle_size - 
    
    ds_size = len(ds) # num of files (images) in dataset
    
    if shuffle:   # shuffle condition
        ds = ds.shuffle(shuffle_size, seed = 12) # shuffle dataset operation
    
    train_size = int(train_split * ds_size) # num of files used for training
    
    train_ds = ds.take(train_size) # get training dataset from full dataset
    
    val_ds = ds.skip(train_size)   # get validation dataset, skip images used for training
    
    return train_ds, val_ds

def prepare_ds(ds):
    # Apply the data augmentation to the train dataset
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    return ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def printScores(model, test_ds):
    # Pring the loss and accracy of the model on a test dataset
    scores = model.evaluate(test_ds)
    loss, acc = scores
    print('\nEvaluation on Test Dataset:')
    print(f'\n\t Loss = {loss}, Accuracy = {acc}, Scores = {scores} \n')

def plotLossAcc(acc, val_acc, loss, val_loss):
    # Plot training and validation accuracy
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(EPOCHS), acc, label='Training Accuracy')
    plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
    # plt.hlines(0.7, 0, 50)
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(EPOCHS), loss, label='Training Loss')
    plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
    # plt.hlines(0.7, 0, 50)
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.show()

def showPrediction(model, test_ds):
    # Demo a prediction on a single image
    for images_batch, labels_batch in test_ds.take(1):
    
        # get an image form the dataset and its correspoding label
        first_image = images_batch[0].numpy().astype('uint8')
        first_label = labels_batch[0].numpy()
        
        actual_label = class_names[first_label]
        
        print("Prediction on a single image:")
        print("actual label:", actual_label)

        batch_prediction = model.predict(images_batch)
        prediction_label = class_names[np.argmax(batch_prediction[0])]
        print("predicted label:", prediction_label)
        plt.imshow(first_image)
        plt.title(f"actual class:{actual_label}, prediction class:{prediction_label}")
        plt.show()

def predict(model, img):
    # Helper function for displayImages
    # Returns prediction class nad confidence of an image
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # create a batch
    
    # Get list of probability for each class
    predictions = model.predict(img_array)[0]
    
    # Determine class of highest probability and the confidence
    max_confidence_index = np.argmax(predictions)
    predicted_class = class_names[max_confidence_index]
    confidence = round(100 * (predictions[max_confidence_index]), 2)
  
    return predicted_class, confidence

def displayImages(model, test_ds):
    # Displays predictions on 6 images
    for images, labels in test_ds.take(1):
        for i in range(6):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype('uint8'))

            predicted_class, confidence = predict(model, images[i].numpy())
            actual_class = class_names[labels[i]]

            plt.title(f"Actual: {actual_class}, \n Predicted: {predicted_class}.\n Confidence: {confidence}%")

            plt.rcParams["figure.figsize"] = (20, 10) # increases image size
            plt.axis("off")
            plt.tight_layout() # if images are too close, separates them 
    plt.show()

def saveModel(model, save_dir, version):
    # Save the trained model
    model.save(f"{save_dir}/ripe{version}")

def saveHist(history, log_dir, version):
    # Save a the loss and accuracy for the model training
    np.save(f'{log_dir}/ripe{version}.npy', history.history)
    
def testPrediction(model, ds):
    # Used for debuging purposes
    for images_batch, labels_batch in ds.take(1):
    
        image = images_batch[0]
    
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0) # create a batch

        predictions = model.predict(img_array)     
        print(predictions)    

#####################################################################

# Define Constants
IMAGE_EDGE_SIZE_X = 256 # Size of the image width 
IMAGE_EDGE_SIZE_Y = 256 # Size of the image height 
IMAGE_SHAPE = (IMAGE_EDGE_SIZE_X, IMAGE_EDGE_SIZE_Y) # image a 2000 by 2000 matrix; (2000, 2000) shape
BATCH_SIZE = 32 # size of batch; how many images are to be loaded into the dataset at a time
CHANNELS = 3    # since image is in RGB, num of channels is 3
EPOCHS = 120 # 50, 150
NUM_CLASSES = 6

## Load data into tensorflow dataset

# Datasets 
TRAIN_DS_DIR = r"datasets\in_use_datasets\ripeness\train_ds_old_oversample"
TEST_DS_DIR = r"datasets\in_use_datasets\ripeness\test_ds_old_oversample"

# Import and load training and validation dataset
train_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DS_DIR,
    shuffle=True,
    image_size = IMAGE_SHAPE,
    batch_size = BATCH_SIZE
)

# Import and load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DS_DIR,
    shuffle=True,
    image_size = IMAGE_SHAPE,
    batch_size = BATCH_SIZE
)

# get class names and store in variable for later use in validation
class_names = train_val_ds.class_names

# See an image from the train or test datasets
# showDsImage(train_val_ds)

# showDsImage(test_ds)

## Divide data for training, validation, and testing sets

# get partitions for training, validation, and testing using partitioning function
train_ds, val_ds = get_dataset_partitions_tf(train_val_ds)

## Some optimization to reduce training time

# optimize dataset so training will run quickly
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

## Preprocessing

# normalize RGB data, and resize images of different size than expected (will be part of the model)
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_EDGE_SIZE_X, IMAGE_EDGE_SIZE_Y),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

# augment data wiht a random horizontal and veritcal flip, and with a random rotation
data_augmentation = tf.keras.Sequential([
    # rescale_train,
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomTranslation(.15, .15, fill_mode='reflect'),
    layers.experimental.preprocessing.RandomZoom(.15, .15, fill_mode='reflect')
])

# Apply Data Augmentation to ds
# train_ds = prepare_ds(train_ds)

## Building the model

# set parameter constants
PARAM_IMAGE_SHAPE = (BATCH_SIZE, IMAGE_EDGE_SIZE_X, IMAGE_EDGE_SIZE_Y, CHANNELS)

# set model parameters
# num of convolutional and pooling layers is determined by trial and error to get highest accuarcy
# num of filters, filter size, and the activation function type are also determined by trail and error to get highest accuarcy

model = models.Sequential([
    resize_and_rescale,  # layer used to resize and reduce the data
    data_augmentation,
    layers.Conv2D(64, (3, 3), activation='relu', input_shape = PARAM_IMAGE_SHAPE[1:]),    # Convoutional layer with relu attached
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)), # pooling layer for 2 by 2 area
    layers.Conv2D(56, (3, 3), activation='relu'),    # Convoutional layer with relu attached
    layers.Conv2D(56, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)), # pooling layer for 2 by 2 area
    layers.Conv2D(48, (3, 3), activation='relu'),
    layers.Conv2D(48, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(),   # layer to flatten the latest feature map
    layers.Dense(32, activation='relu'),    # use a dense layer to for classification
    layers.Dense(NUM_CLASSES, activation='softmax'),    # use softmax activation function to normalize probability of classes
])

# model = models.Sequential([
    # resize_and_rescale,  # layer used to resize and reduce the data
    # data_augmentation,  # layer that implements data agumentation
    # layers.Conv2D(30, (3, 3), activation='relu', input_shape = PARAM_IMAGE_SHAPE[1:]),    # Convoutional layer wiht relu attached
    # layers.Conv2D(30, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)), # pooling layer for 2 by 2 area
    # layers.Conv2D(40, (3, 3), activation='relu'),
    # layers.Conv2D(40, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(50, (3, 3), activation='relu'),
    # layers.Conv2D(50, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(60, (3, 3), activation='relu'),
    # layers.Conv2D(60, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),
    # layers.Dropout(0.2),
    # layers.Flatten(),   # layer to flatten the latest feature map
    # layers.Dense(60, activation='relu'),    # use a dense layer to for classification
    # layers.Dense(NUM_CLASSES, activation='softmax'),    # use softmax activation function to normalize probability of classes
# ])  # Note, the resize_and_rescale and the data_augmentation layers are part of the main model (will not go away after training)


# use .build to create model
model.build(input_shape=PARAM_IMAGE_SHAPE)

# Overview of model architecture
model.summary()

# Compile the model to add optimizers
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

## Now train the model using fit

# model is named history for evaluation after training
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)

## Model Evaluation

# Quick evaluation of the Model
printScores(model, test_ds)

# Use history for more indepth evaluation

# Store accuracy and loss in variables for graphical evaluation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot Accuracy and Loss vs. Epocs
plotLossAcc(acc, val_acc, loss, val_loss)


##########################################################################################

# Make a prediction from a single image (just for quick demonstrations of model)
# showPrediction(model, test_ds)

##########################################################################################
  
## Demonstrate model predictions on multiple images
displayImages(model, test_ds)
    
##########################################################################################

## Save the model to a directory to use later

# Set dir to save model, history log, and set version
MODEL_SAVE_DIR = r'saved_models\ripeness'
MODEL_HIST_DIR = r'saved_models\ripeness\model_history_logs'
model_version = '_old_oversample_80'

saveModel(model, MODEL_SAVE_DIR, model_version)

saveHist(history, MODEL_HIST_DIR, model_version)

##########################################################################################

## Debug predictions confidence > 100%
# testPrediction(model, test_ds)
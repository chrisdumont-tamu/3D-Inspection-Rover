# import ML and Visualization libraries
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

#####################################################################
## Functions 
def showDsImage(dataset):
    for image_batch, label_batch in dataset.take(1):
        plt.imshow(image_batch[0].numpy().astype("uint8"))
        plt.show()

def get_dataset_partitions_tf(ds, train_split=0.85, shuffle=True, shuffle_size=10000):
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
    model.save(f"{save_dir}/defect{version}")

def saveHist(history, log_dir, version):
    # Save a the loss and accuracy for the model training
    np.save(f'{log_dir}/defect{version}.npy', history.history)
    
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
BATCH_SIZE = 24 # size of batch; how many images are to be loaded into the dataset at a time
CHANNELS = 3    # since image is in RGB, num of channels is 3
EPOCHS = 150
NUM_CLASSES = 4

## Load data into tensorflow dataset

# Datasets with and w/out inlcuding defective class (for bugfix purposes)
TRAIN_DS_DIR = r"datasets\in_use_datasets\defective\train_ds"
TEST_DS_DIR = r"datasets\in_use_datasets\defective\test_ds"

# Import and load training and validation dataset
train_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DS_DIR,
    shuffle=True,
    image_size = IMAGE_SHAPE,
    batch_size = BATCH_SIZE
)

# Impoert and load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DS_DIR,
    shuffle=True,
    image_size = IMAGE_SHAPE,
    batch_size = BATCH_SIZE
)

# print(f'\n\ntrain ds len: {len(train_val_ds)}\n\n')
# showDsImage(train_val_ds)
# image_batch, classes_batch = train_val_ds

for image_batch, label_batch in train_val_ds.take(1):
        # plt.imshow(image_batch[0].numpy().astype("uint8"))
        # plt.show()
        print(f'label_batch: {label_batch}')

# print(f'image_batch shape: {np.shape(image_batch)}')
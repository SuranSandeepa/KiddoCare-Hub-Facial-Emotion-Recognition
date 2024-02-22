#system operations
import sys, os
#data manipulation
import pandas as pd
#numerical operations
import numpy as np

# Import Keras libraries for building and training the model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

# Read the dataset from a CSV file
df=pd.read_csv('fer2013.csv')

# print(df.info())
# print(df["Usage"].value_counts())

# print(df.head())
# Initialize lists to store training and testing data along with labels
X_train,train_y,X_test,test_y=[],[],[],[]

# Loop through the dataset and extract pixel values and labels
for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
           X_train.append(np.array(val,'float32'))
           train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
           X_test.append(np.array(val,'float32'))
           test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")

# Define some constants for the model
num_features = 64
num_labels = 7
batch_size = 64
epochs = 30
width, height = 48, 48

# Convert data to NumPy arrays and perform one-hot encoding on labels
# X_train = np.array(X_train, 'float32')
# X_train = np.array(X_train,'float32')
# train_y = np.array(train_y,'float32')
# X_test = np.array(X_test,'float32')
# test_y = np.array(test_y,'float32')

# train_y = to_categorical(train_y, num_classes=num_labels)
# test_y = to_categorical(test_y, num_classes=num_labels)

# Convert data to NumPy arrays and perform one-hot encoding on labels
X_train = np.array(X_train, 'float32')
train_y = np.array(train_y, 'float32')  # Convert emotion labels to NumPy array
# Convert emotion labels to integer (required for Keras)
train_y_int = np.array(train_y, 'int')
# One-hot encode the integer labels
train_y_one_hot = to_categorical(train_y_int, num_classes=num_labels)
X_test = np.array(X_test, 'float32')
test_y = np.array(test_y, 'float32')  # Convert emotion labels to NumPy array
# Convert emotion labels to integer (required for Keras)
test_y_int = np.array(test_y, 'int')
# One-hot encode the integer labels
test_y_one_hot = to_categorical(test_y_int, num_classes=num_labels)


# Normalize the data
#cannot produce
#normalizing data between oand 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

# Reshape the data for the Convolutional Neural Network
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

# Create a Sequential model for the CNN
# print(f"shape:{X_train.shape}")
##designing the cnn
#1st convolution layer
model = Sequential()

# Define the architecture of the CNN
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels, activation='softmax'))

# model.summary()

#Compliling the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

# #Training the model
# model.fit(X_train, train_y,
#             batch_size=batch_size,
#             epochs=epochs,
#             verbose=1,
#             validation_data=(X_test, test_y),
#             shuffle=True)

#Training the model
history = model.fit(X_train, train_y_one_hot,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, test_y_one_hot),
                    shuffle=True)

# Print model summary
model.summary()


# Plot training & validation accuracy values
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#Saving the  model to  use it later on
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")


# Use the trained model to make predictions on the test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_y_one_hot, axis=1) 

# Calculate confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score

conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate accuracy percentage
accuracy_percentage = accuracy_score(y_true, y_pred_classes) * 100
print(f"Accuracy Percentage: {accuracy_percentage:.2f}%")
import sys
sys.path.append('/Users/anekha/Documents/GitHub/jewelrecs/')

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from models.features import X_train, y_train, X_val, y_val, X_test, y_test
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Training complete for epoch {epoch+1}")
        print(f"Loss: {logs.get('loss')}")
        print(f"Accuracy: {logs.get('accuracy')}")
        print(f"Validation Loss: {logs.get('val_loss')}")
        print(f"Validation Accuracy: {logs.get('val_accuracy')}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define a learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Define and compile the model with modifications
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(np.unique(y_train)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Instantiate the custom callback
custom_callback = CustomCallback()

# Define the learning rate scheduler callback
lr_scheduler = LearningRateScheduler(scheduler)

# Train the model with the custom callback and learning rate scheduler
history = model.fit(X_train_scaled, y_train,
                    epochs=500,
                    batch_size=64,
                    validation_data=(X_val_scaled, y_val),
                    callbacks=[custom_callback, lr_scheduler])

# Evaluate the model on the test set
X_test_scaled = scaler.transform(X_test)
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the model
model.save('face_shape_model_improved.h5')

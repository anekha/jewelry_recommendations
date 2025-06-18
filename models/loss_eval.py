import matplotlib.pyplot as plt

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, X_test, y_test):
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.train_loss.append(logs.get('loss'))

        # Evaluate the model on the validation set
        val_loss = self.model.evaluate(self.X_val, self.y_val, verbose=0)[0]
        self.val_loss.append(val_loss)

        # Evaluate the model on the test set
        test_loss = self.model.evaluate(self.X_test, self.y_test, verbose=0)[0]
        self.test_loss.append(test_loss)

# Assuming you have trained the model with the custom callback as before
# Now you can plot the losses

# Plot the training, validation, and test losses
plt.figure(figsize=(10, 6))
epochs = range(1, len(custom_callback.train_loss) + 1)
plt.plot(epochs, custom_callback.train_loss, 'b', label='Training loss')
plt.plot(epochs, custom_callback.val_loss, 'g', label='Validation loss')
plt.plot(epochs, custom_callback.test_loss, 'r', label='Test loss')
plt.title('Training, Validation, and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

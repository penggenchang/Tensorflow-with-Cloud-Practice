import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
## so if the parent folder is in th module search path, it sould be OK
from models.cnn_model import create_model
from data.loader import load_cifar10
import tensorflow as tf
import os

def train_model(epochs=10, batch_size=16):
    # 1. Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # 2. Create model
    model = create_model()

    # 3. Compile model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. CSV logger to save performance metrics per epoch
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(log_dir,'training_log.csv'))


    # 5. Train model
    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              callbacks=[csv_logger])

    # 5. Save model
    model.save("cnn_cifar10_model.h5")
    print("✅ Model saved to cnn_cifar10_model.h5")
    print("Logs in logs/training_log.csv")

# 🔍 Test block
if __name__ == "__main__":
    try:
        train_model(epochs=10)
        print("✅ Training finished successfully.")
    except Exception as e:
        print("❌ An error occurred during training:")
        print(e)


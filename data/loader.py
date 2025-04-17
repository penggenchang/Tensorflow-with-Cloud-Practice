import tensorflow as tf
import os

def load_cifar10(data_dir=None):
    """
    Load CIFAR-10 dataset from local cache or download if not found.
    Optionally specify a custom data_dir (useful for Docker).
    """
    if data_dir is None:
        # Use default Keras directory (~/.keras/datasets)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
        # Make sure directory exists
        os.makedirs(data_dir, exist_ok=True)
        # Set the environment variable to control where datasets are stored
        os.environ['TFDS_DATA_DIR'] = data_dir
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":   ## only run it as a direct file, rather than being imported
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    print("load good!")
    print(f"training set: x={x_train.shape}, y={y_train.shape}")
    print(f"x dtype: {x_train.dtype}, range: [{x_train.min()}, {x_train.max()}]")
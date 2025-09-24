import tensorflow as tf
from keras.applications.xception import Xception

try:
    # Try normal GPU execution
    model = Xception(include_top=False, pooling='avg', weights='imagenet')
except tf.errors.UnimplementedError:
    print("GPU CuDNN mismatch detected. Falling back to CPU...")
    with tf.device('/CPU:0'):
        model = Xception(include_top=False, pooling='avg', weights='imagenet')

import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description="Convert Keras model to tflite")
parser.add_argument('--input-model', help='Keras model file name', type=str)
parser.add_argument('--output-model', help='Output tflite model name', type=str)
args = parser.parse_args()

if args.input_model is None or args.output_model is None:
    print("Please specify all required arguments (Input model and Output model names)")
    exit(1)

model = tf.keras.models.load_model( args.input_model +'.keras')
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

with open( args.output_model + '.tflite', 'wb') as f:
    f.write(tflite_model)
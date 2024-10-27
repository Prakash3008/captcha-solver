#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import argparse
import tensorflow as tf
import tensorflow.keras as keras
import time

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    decoded_string = ''.join([characters[x] for x in y])
    return decoded_string.rstrip('-')

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()


    

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    with tf.device('/cpu:0'):
        with open(args.output, 'w') as output_file:
            json_file = open(args.model_name+'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            model.load_weights(args.model_name + '.keras')
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                          metrics=['accuracy'])

            for x in os.listdir(args.captcha_dir):
                raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
                
                gray_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY)
                
                blurred_data = cv2.GaussianBlur(gray_data, (5, 5), 0)
                
                binary_data = cv2.adaptiveThreshold(blurred_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY_INV, 11, 2)
                
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                morphed_data = cv2.morphologyEx(binary_data, cv2.MORPH_OPEN, kernel)
                
                morphed_data = cv2.dilate(morphed_data, kernel, iterations=1)
                
                rgb_data = cv2.cvtColor(morphed_data, cv2.COLOR_GRAY2RGB)

                image = numpy.array(rgb_data) / 255.0

                (c, h, w) = image.shape
                image = image.reshape([-1, c, h, w])

                # Make prediction using the model
                prediction = model.predict(image)

                decoded_preds = decode(prediction, captcha_symbols)

                # Write the prediction to the output file
                output_file.write(x + "," + decoded_preds[0] + "\n")
                
                # Print the classified image and the model's prediction
                print('Classified: ' + x)
                print('The model predicts: ' + decoded_preds[0])

    end_time = time.time()
    time_taken = end_time - start_time
    with open('time_log/classification_time.txt', 'w') as file:
        file.write('Classification: {:.2f} seconds'.format(time_taken))
        print('Time taken for classifying all the captchas:', time_taken, 'seconds')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import argparse
import tflite_runtime.interpreter as tflite
import time



def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])


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

    with open(args.output, 'w') as output_file:
        interpreter = tflite.Interpreter(model_path=args.model_name+'.tflite')
        interpreter.allocate_tensors()

        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()
        for x in sorted(os.listdir(args.captcha_dir)):
            # load image and preprocess it
            raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            image = numpy.array(rgb_data) / 255.0
            (c, h, w) = image.shape
            image = image.reshape([-1, c, h, w]).astype('float32')
            interpreter.set_tensor(inputs[0]['index'], image)
            interpreter.invoke()
            output_list = []
            for i in range(6):
                output_data = interpreter.get_tensor(outputs[i]['index'])
                output_list.append(output_data)
            output_file.write(x + ", " + decode(captcha_symbols, output_list) + "\n")
            print('Classified ' + x)

    end_time = time.time()
    time_taken = end_time - start_time
    with open('time_log/classification_time_pi.txt', 'w') as file:
        file.write('Classification: {:.2f} seconds'.format(time_taken))
        print('Time taken for classifying all the captchas:', time_taken, 'seconds')


if __name__ == '__main__':
    main()
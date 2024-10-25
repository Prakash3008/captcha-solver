#!/usr/bin/env python3

import os
import numpy
import random
import cv2
import argparse
import captcha.image
import time

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--count', help='How many captchas to generate', type=int)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--fonts', help='File with the list of fonts to use in captchas', type=str)
    args = parser.parse_args()

    if args.width is None or args.height is None or args.count is None or args.output_dir is None or args.symbols is None or args.fonts is None:
        print("Please specify all required arguments (width, height, count, output_dir, symbols, fonts)")
        exit(1)

    with open(args.fonts, 'r') as font_file:
        font_list = [line.strip() for line in font_file if line.strip()]

    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    train_dir = os.path.join(args.output_dir, 'train')
    test_dir = os.path.join(args.output_dir, 'validate')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    num_fonts = len(font_list)
    captchas_per_font = args.count // num_fonts
    extra_captchas = args.count % num_fonts  

    print(f"Generating {captchas_per_font} captchas per font, with {extra_captchas} extra captchas distributed.")

    captcha_idx = 0
    for font in font_list:
        captchas_to_generate = captchas_per_font + (1 if captcha_idx < extra_captchas else 0)
        captcha_idx += 1

        print(f"Generating {captchas_to_generate} captchas with font: {font}")
        
        captcha_generator = captcha.image.ImageCaptcha(width=args.width, height=args.height, fonts=[font])

        for i in range(captchas_to_generate):
            random_length = random.randint(1, 8)
            random_str = ''.join([random.choice(captcha_symbols) for j in range(random_length)])
            image_path = os.path.join(train_dir if i < int(captchas_to_generate * 0.9) else test_dir, random_str + '.png')

            if os.path.exists(image_path):
                continue

            image = numpy.array(captcha_generator.generate_image(random_str))
            cv2.imwrite(image_path, image)

    end_time = time.time()
    time_taken = end_time - start_time
    with open('time_log/generation_time.txt', 'w') as file:
        file.write('Generation of {} Captchas: {:.2f} seconds'.format(args.count, time_taken))
        print('Time taken for generating captchas:', time_taken, 'seconds')

if __name__ == '__main__':
    main()

# captcha-solver

The repository contains code to solve captchas with different lengths, fonts and great amount of noise in them.
This project is done for the course module Scalable Computing - CS7NS1.

### Team Number: 30

### Members:
1. Prakash Narasimhan
2. Evanka Dsouza

# Steps to run the code

1. Make sure to install Python version <= 3.9
2. Install all the required packages from the requirements file
3. Download the captchas for a specific user from the server
4. Generate training captchas
5. Train the CNN captcha model with the generated training data
6. Convert the .keras model to .tflite model to classify it on the 32-bit Raspberry PI
7. Classify the downloaded captchas using the trained model
8. Sort the output csv

## Step 1 & 2: Installing the required packages

```python

pip3 install -r requirements.txt

```
> This installs the required packages: requests, pillow, numpy, captcha, tensorflow, tflite, opencv-python

## Step 3: Download the captchas

> Use the following command with the required arguments to download the captchas

```python

python3 get_captchas.py --short-username shortname --output-folder-name user/downloaded_captchas --file-list-name user/filelist.txt

```


## Step 4: Generate training captchas for the model

> Use the following command with the required arguments to generate the captchas 

```python

python3 generate.py --width 192 --height 96 --count 220000 --output-dir user/training --symbols symbols.txt --fonts user/fonts.txt

```

> The above command generates captchas of equal amount for the number of fonts provided and split the captchas into train set and validate set.

## Step 5: Train the model

> Use the following command with the required arguments to train the model

```python

python3 train.py --width 192 --height 96 --length 7 --batch-size 64 --train-dataset user/training/train --validate-dataset user/training/validate --output-model-name user/captcha_model --epochs 12 --symbols symbols.txt

```

## Step 6: Convert the keras model to tflite

> Use the following command with the required arguments to convert the model

```python

python3 convert.py --input-model user/captcha_model --output-model user/captcha_model 

```

> To use the model on the Raspberry PI to classify the captchas

## Step 7: Classification of the captchas 64-bit devices

> Use the following command with the required arguments for classification on 64-bit devices

```python

python3 classify.py --model-name user/captcha_model --captcha-dir user/captchas --output user/classify_output.csv --symbols symbols.txt --model-type tflite

```

> To use the model on the Raspberry PI to classify the captchas

## Step 8: Sort the resultant CSV file

> Use the following command with the required arguments for sorting the csv file alphabetically


```python

python3 sort.py --input-file user/classify_output.csv --output-file user/sorted_classify_output.csv

```

> After sorting the csv file, add the short username in the first line of the csv file and then save it


## Step 9: Running Classification of captchas on 32-bit devices

> Use the following command with the required arguments for classification on 32-bit devices


```python

python3 classify_tflite.py --model-name user/captcha_model --captcha-dir user/captchas --output user/tflite_classify_output.csv --symbols symbols.txt 

```
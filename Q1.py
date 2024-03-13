import pandas as pd
import numpy as np
import os
import requests
import random
import cv2
import glob

import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the pre-trained MobileNet model
mobilenet_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def extract_features_from_images(image_folder):
    features = []
    image_paths = os.listdir(image_folder)
    
    for image_path in image_paths:
        image = load_img(os.path.join(image_folder, image_path), target_size=(224, 224))
        image_array = img_to_array(image)
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)
        features.append(mobilenet_model.predict(image_array))
    
    return np.array(features)

def resize_images(input_folder, output_folder, new_width, new_height):
	os.makedirs(output_folder, exist_ok=True)

	for filename in glob.glob(os.path.join(input_folder, '*.jpg')):
		img = cv2.imread(filename)
		resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
		
		output_filename = output_folder + '/' + (filename.split('/')[-1])[:-4] + '_resized.jpg'
		cv2.imwrite(output_filename, resized_image)
	print("Images Resized Succesfully")

def adjust_brightness(input_folder, output_folder, brightness_factor=100.0):
    os.makedirs(output_folder, exist_ok=True)

    for filename in glob.glob(os.path.join(input_folder, '*.jpg')):
        img = cv2.imread(filename)
        # Convert to HSV (hue, saturation, value) color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Adjust brightness
        v = v.astype('float64')  # Convert to float to prevent data loss
        v += brightness_factor  # Add the brightness factor
        v = np.clip(v, 0, 255)  # Ensure the values are within [0, 255]
        v = v.astype('uint8')  # Convert back to uint8

        final_hsv = cv2.merge((h, s, v))
        brightened_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        # Save the brightened image
        output_filename = os.path.join(output_folder, os.path.basename(filename)[:-4] + '_brightened.jpg')
        cv2.imwrite(output_filename, brightened_image)

    print("Brightness adjusted for all images successfully.")

def adjust_exposure(input_folder, output_folder, exposure_factor=100.0):
    os.makedirs(output_folder, exist_ok=True)

    for filename in glob.glob(os.path.join(input_folder, '*.jpg')):
        img = cv2.imread(filename)
        # Convert to HSV (hue, saturation, value) color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Adjust exposure
        v = v.astype('float64')  # Convert to float to prevent data loss
        v *= exposure_factor  # Multiply by the exposure factor
        v = np.clip(v, 0, 255)  # Ensure the values are within [0, 255]
        v = v.astype('uint8')  # Convert back to uint8

        final_hsv = cv2.merge((h, s, v))
        exposed_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        # Save the exposed image
        output_filename = os.path.join(output_folder, os.path.basename(filename)[:-4] + '_exposed.jpg')
        cv2.imwrite(output_filename, exposed_image)

    print("Exposure adjusted for all images successfully.")

def random_flip_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in glob.glob(os.path.join(input_folder, '*.jpg')):
        img = cv2.imread(filename)
        
        # Randomly decide the flip code: 0 for vertical, 1 for horizontal, -1 for both axes
        flip_code = random.choice([0, 1, -1])
        flipped_image = cv2.flip(img, flip_code)

        # Save the flipped image
        output_filename = os.path.join(output_folder, os.path.basename(filename)[:-4] + '_flipped.jpg')
        cv2.imwrite(output_filename, flipped_image)

    print("Random flips applied to all images successfully.")

def blur_images(input_folder, output_folder, blur_strength=(5, 5)):
    os.makedirs(output_folder, exist_ok=True)

    for filename in glob.glob(os.path.join(input_folder, '*.jpg')):
        img = cv2.imread(filename)
        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(img, blur_strength, 0)

        # Save the blurred image
        output_filename = os.path.join(output_folder, os.path.basename(filename)[:-4] + '_blurred.jpg')
        cv2.imwrite(output_filename, blurred_image)

    print("Gaussian blur applied to all images successfully.")

def rotate_images(input_folder, output_folder, rotation_angle):

    os.makedirs(output_folder, exist_ok=True)
    for filename in glob.glob(os.path.join(input_folder, '*.jpg')):
        img = cv2.imread(filename)

        if rotation_angle == 90:
            rotated_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            rotated_image = cv2.rotate(img, cv2.ROTATE_180)
        elif rotation_angle == 270:
            rotated_image = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            print(f"Invalid rotation angle: {rotation_angle}. Skipping {filename}.")
            continue

        output_filename = output_folder + '/' + (filename.split('/')[-1])[:-4] + '_{}_rotated.jpg'.format(rotation_angle)
        cv2.imwrite(output_filename, rotated_image)
    print("Images Rotated Succesfully")    

def Download_img_CSV(file_path,folder_path):
	os.makedirs(folder_path, exist_ok=True)

	df = pd.read_csv(file_path)	
	for index, row in df.iterrows():
	    uid = row["Unnamed: 0"]
	    images = row["Image"]
	    text = row["Review Text"]
	    
	    count = 0
	    for image in images[1:-1].split(','):
	    	url = image.strip()[1:-1]
	    	response = requests.get(url)
	    	if response.status_code == 200:
	    		with open(folder_path+'/{}_{}.jpg'.format(uid,count), 'wb') as f:
	    			f.write(response.content)
	    			count+= 1
	print("Images Downloaded Succesfully")


csv_path = os.getcwd() + '/A2_Data.csv'
download_path = os.getcwd() + "/Download_Images"
input_path = os.getcwd() + "/Input_Images"
preprocessed_path = os.getcwd() + "/Preprocessed_Images"


Download_img_CSV(csv_path,download_path)
resize_images(download_path,input_path,300 ,300)
resize_images(download_path,preprocessed_path,300 ,300)

rotate_images(input_path, preprocessed_path, 90)
# rotate_images(input_path, preprocessed_path, 180)
# rotate_images(input_path, preprocessed_path, 270)

adjust_brightness(input_path, preprocessed_path)
adjust_exposure(input_path, preprocessed_path)
random_flip_images(input_path, preprocessed_path)
blur_images(input_path, preprocessed_path)

# Extract features from the images
image_features = extract_features_from_images(preprocessed_path)

# Print the shape of the extracted features
print(f"Extracted features shape: {image_features.shape}")

with open('features.pkl', 'wb') as file:
    pickle.dump(image_features, file)

print("The array has been stored in pickle format as 'features.pkl'.")
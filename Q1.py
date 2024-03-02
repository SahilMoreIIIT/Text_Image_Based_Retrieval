import pandas as pd
import os
import requests
import cv2
import glob

def resize_images(input_folder, output_folder, new_width, new_height):
	os.makedirs(output_folder, exist_ok=True)

	for filename in glob.glob(os.path.join(input_folder, '*.jpg')):
		img = cv2.imread(filename)
		resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
		
		output_filename = output_folder + '/' + (filename.split('/')[-1])[:-4] + '_resized.jpg'
		cv2.imwrite(output_filename, resized_image)
	print("Images Resized Succesfully")


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


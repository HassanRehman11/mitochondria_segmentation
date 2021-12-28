# Mitochondria Segmentation

Semester project based on mitochondria segmentation from the electron microscopy image dataset

## Conversion of Data Set
>> Open the file DataConversion_Prepration.ipynb
>> Original Data set is in TIF file resides in actual_data folder
>> Run all the commands and data will be populated accordingly for training and testing

## Training Dataset
>> Open the DataTraining.ipynb file
>> To tweek the parameters change the config.py file 
>> Config.py should be change before data set prepration to avoid any errors
>> Run all the cells

## Testing Dataset
>> Run DataTesting.ipynb
>> It will open all the files from testing_data that resides in DATA folder
>> It will generate the original, original mask and predicted mask in horizontal stack and store in results folder
>> Also provides you the testing IOU score

Following data will be generated:



## Inference.py
>> run "python3 inference.py DATA/test_data/images/test-0081.png"
>> This will give overlay image of mask 

Following data will be generated:

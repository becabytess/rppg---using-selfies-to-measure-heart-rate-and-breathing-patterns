import kagglehub
import shutil

path = kagglehub.dataset_download("malekdinarito/ubfc-rppg-dataset")

#move the dataset to the data folder in the main directory 

shutil.move(path, "data/UBFC-RPPG-Dataset")



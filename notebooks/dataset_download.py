import os
from kaggle.api.kaggle_api_extended import KaggleApi
from tkinter import messagebox

def download_datasets(dataset_names):
    try:
        os.environ["KAGGLE_USERNAME"] = "bokkadurgaprasad"
        os.environ["KAGGLE_KEY"] = "e3ee76ab1475679e50e1d96586eef304"

        dataset_path = "./Dataset"

        api = KaggleApi()
        api.authenticate()


        for name in dataset_names:
            api.dataset_download_files(name, path=dataset_path, unzip=True)
            print(f"Dataset {dataset_names.index(name) + 1} downloaded and extracted successfully.")
        messagebox.showinfo("Success", "Datasets downloaded successfully!")
        
    except Exception as e:
        messagebox.showerror("Error", "Please check your internet connection.")


dataset_names = ["rohitupadhya/256objects", "paultimothymooney/chest-xray-pneumonia"]
download_datasets(dataset_names)

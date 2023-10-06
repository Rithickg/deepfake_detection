import os
import pandas as pd

def create_dataframe(image_folder, label):
    # Get the list of image file paths in the folder
    image_files = os.listdir(image_folder)
    image_paths = [os.path.join(image_folder, file) for file in image_files]

    # Create a list of image names by extracting the filename without the directory path and extension
    image_names = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]

    # Create a list of corresponding labels
    labels = [label] * len(image_paths)

    # Create a DataFrame with 'filename', 'image_name', 'label', 'binary_label', and 'file_path' columns
    df = pd.DataFrame({'filename': image_files, 'image_name': image_names, 'label': labels, 'file_path': image_paths})

    return df

if __name__ == "__main__":
    # Replace 'path_to_real_folder' and 'path_to_deepfake_folder' with the actual paths to your image folders
    real_folder = '/content/drive/MyDrive/Colab_Notebooks/Deepfake_Detection/deepfake_database/train/real'
    deepfake_folder = '/content/drive/MyDrive/Colab_Notebooks/Deepfake_Detection/deepfake_database/train/df'

    # Create DataFrames for real and deepfake images
    real_df = create_dataframe(real_folder, 'real')
    deepfake_df = create_dataframe(deepfake_folder, 'deepfake')

    # Combine the DataFrames for real and deepfake images
    combined_df = pd.concat([real_df, deepfake_df], ignore_index=True)

    # Create a binary label column where 'real' is labeled as 0 and 'deepfake' is labeled as 1
    combined_df['binary_label'] = combined_df['label'].apply(lambda x: 0 if x == 'real' else 1)

    # Optionally, shuffle the dataset
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)

    # Display the first few rows of the combined DataFrame
    print(combined_df.head())

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv('/content/drive/MyDrive/Colab_Notebooks/Deepfake_Detection/both_Label_dataset.csv', index=False)

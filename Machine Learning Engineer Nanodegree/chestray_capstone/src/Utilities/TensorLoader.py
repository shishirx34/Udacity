from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
from Configurations.Config import Config

config = Config.GetConfig()

IMAGE_SIZE = (config.ImageSizeWidth, config.ImageSizeHeight)

image_data_generator = ImageDataGenerator(
    samplewise_center=True, 
    samplewise_std_normalization=True,
    rotation_range = 4,             # minor rotation twists
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    brightness_range = [0.75, 1.25],   # 75% to 125% brightness
    fill_mode = "nearest",
    horizontal_flip=False,
    vertical_flip=False,
    zoom_range=0.10                 # 10% zoom
)

def Load_Data_Images_For_DataFrame(input_df, path_column, prediction_column, batch_size, mode):
    directory = config.DataFolderPath
    generated_df = image_data_generator.flow_from_directory(directory,
        class_mode='sparse',
        target_size = IMAGE_SIZE,
        color_mode = mode,
        batch_size = batch_size)
    generated_df.classes = np.stack(input_df[prediction_column].values)
    generated_df.samples = input_df.shape[0]
    generated_df.n = input_df.shape[0]
    generated_df._set_index_array()
    generated_df.directory = directory
    print("Generated data frame with loaded images: ", input_df.shape)
    return generated_df
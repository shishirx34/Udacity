# ChestR - Convolutional Neural Network for Chest X-RAY Disesase Prediction

This project contains the source code for the convolutional neural network developed for Udacity's Machine learning engineering capstone project.

## Requirements
These are the software specifications based on which this source code was developed. It is still possible to compile for different specs the code might need to be tweaked.
- Python 3.5
- Tensorflow GPU 1.5
- CUDA Toolkit 9.0
- CuDNN 7.0.4
- Windows 10 - Hosted on Azure VM - NC6 Tier with 7.5Gb Nvidia Tesla M60 Graphics card.
- See the python-modules.txt to see the installed modules for running this project.

This data is based off of NIH Chest X-Ray dataset. This data is required to train and test the developed models in this project. The dataset is available to be downloaded on  [Kaggle's NIH Chest X-RAY dataset](https://www.kaggle.com/nih-chest-xrays/data/downloads/data.zip/3). Please download and extract all the images folder in following format to ensure correct data load for running models.

```
C:\ChestXray\Image\data\Data_Entry_2017.csv
C:\ChestXray\Image\data\images_*\images\*.jpg
```

## Execution

After cloning this project, load up the `chestray_capstone` project folder in your favorite IDE.

### Configuration

Configure the data folder for the dataset downloaded from the above said link for NIH xrays, by setting the `DataFolderPath` value in `src\Configurations\Config.json` file. Further fine tuning of parameters for training the CNN/MLP can be found in the same file.

### Data Exploration

Run the following file for running data exploration on the dataset from the root of this project: `.\chestray_capstone`

```python
python data-exploration.py
```

This python script will generate the relevant images exploring the data from the dataset in `src\Generated\data-exploration-images` folder.

### Preprocessing

`preprocessor.py` contains the code for pre-processing the data before the model is trained. In this script we get rid of all the features apart from the image file names. Also, as part of preprocessing stage we load the image data and sampled by weights.

### Training Models

As part of the getting disease predictions I developed and trained several neural networks. 

1. Base Multi-layer Perceptron model. To execute the training on this model you can execute following command

    ```
    python src\base-mlp-neural-net.py 16 2
    ```
    First parameter is the number of nodes for the input layer and doubling nodes for subsequent layers. Second parameter is number of epochs for which you wish to intend to train this model.

    This will output the best saved models with prefix `base-mlp-model*` in `Generated\saved-models` folder.

2. Vanilla Convolutional Neural Network i.e. without transfer learning. You can run this model by executing following command

    ```
    python src\chestr-cnn.py 10
    ```
    The parameter is the number of epochs to train the model. 
    
    This will output the best saved models in the `Generated\saved-models` folder with the prefix `chestr-cnn-model*` along with the test data for analysis.

3. Transfer learning with InceptionV3 model.

    ```
    python src\chestr-inceptionv3-cnn.py 10
    ```
    The parameter is the number of epochs to train the model. 
    
    This will output the best saved models in the `Generated\saved-models` folder with the prefix `chestr-inceptv3-model*` along with the test data for analysis.

4. Transfer learning with VGG19 model.

    ```
    python src\chestr-vgg19-cnn.py 10
    ```
    The parameter is the number of epochs to train the model. 
    
    This will output the best saved models in the `Generated\saved-models` folder with the prefix `chestr-vgg19-model*` along with the test data for analysis.


### Analyzing the models

Modify the `src\analyze-neural-net.py` file by setting the appropriate values for the variables `best_model_file_path` and `model_testdata_file_path` for the above generated files by the training models respectively. Run the following command to get the analysis of the models

```
python src\analyze-neural-net.py
```

This script will load the weights and analyze the predictions from the models against the stored test data, it will generate the classificaiton matrix for each of the 13 prediction diseases along with the ROC - AUC for the model and generate sample images from the predictions.
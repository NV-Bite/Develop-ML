# **Notes on Building a Machine Learning Project**

This document explains the steps and insights involved in building a machine learning project. It includes details from preparing the dataset to training and testing the model.
## Installation
Ensure you have the following dependencies installed:

- Python 3.8+
- TensorFlow 2.9+
- Pandas
- NumPy
- Seaborn
- Matplotlib
- tqdm

Install dependencies via pip:
```bash
pip install tensorflow pandas numpy seaborn matplotlib tqdm
```
## **Setup and Imports**
- Environment Cleanup: Removes temporary files to ensure a clean slate for training.
- **Library Imports**: Loads necessary libraries for data manipulation, visualization, and model building.


## **Dataset**

The dataset can be found here:  
[Dataset Link](https://www.kaggle.com/datasets/rizkyyk/dataset-food-classification/data)

  
## **Organizing the Dataset**  

The dataset is organized by renaming and formatting image files in their folders. The process includes:  
1. Renaming files to match their folder names and adding numbers (e.g., `cat_001.jpg`).  
2. Converting unsupported image formats to `.jpg`.  
3. Removing original files after renaming.  

The code used for this process is in: [`organize_image.py`](https://github.com/NV-Bite/Develop-ML/blob/main/image_classification/organize_image.py)   

## Split Dataset

available at [`data_splitting.ipynb`](https://github.com/NV-Bite/Develop-ML/blob/main/image_classification/split_dataset.ipynb)

the output is 3 folders = `train, test, and valid`

---

## **Dataset Preparation**

### **Loading an Image**

This function loads an image from the given path and preprocesses it by resizing and normalizing it.

```python
def load_image(image_path: str) -> tf.Tensor:
    """
    Loads and preprocesses an image from the specified path.

    Parameters:
    - image_path (str): The file path to the image.

    Returns:
    - tf.Tensor: The processed image as a TensorFlow tensor.
    """
    # Ensure the image path exists
    assert os.path.exists(image_path), f'Image path does not exist: {image_path}'
    
    # Read the image file
    image = tf.io.read_file(image_path)
    
    # Decode the image as JPEG or PNG
    try:
        image = tfi.decode_jpeg(image, channels=3)
    except:
        image = tfi.decode_png(image, channels=3)
    
    # Normalize pixel values to the range [0, 1]
    image = tfi.convert_image_dtype(image, tf.float32)
    
    # Resize the image to the desired dimensions
    image = tfi.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Ensure the image tensor has the correct data type
    image = tf.cast(image, tf.float32)
    
    return image
```

### **Loading the Dataset**

This function loads images and their labels into arrays for training, testing, or validation.

```python
def load_dataset(root_path: str, class_names: list, trim: int=None) -> Tuple[np.ndarray, np.ndarray]:
    if trim:
        # Trim the size of the data
        n_samples = len(class_names) * trim
    else:
        # Collect total number of data samples
        n_samples = sum([len(os.listdir(os.path.join(root_path, name))) for name in class_names])

    # Create arrays to store images and labels
    images = np.empty(shape=(n_samples, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    labels = np.empty(shape=(n_samples, 1), dtype=np.int32)

    # Loop over all the image file paths, load and store the images with respective labels
    n_image = 0
    for class_name in tqdm(class_names, desc="Loading"):
        class_path = os.path.join(root_path, class_name)
        image_paths = list(glob(os.path.join(class_path, "*")))[:trim]
        for file_path in image_paths:
            # Load the image
            image = load_image(file_path)

            # Assign label
            label = class_names.index(class_name)

            # Store the image and the respective label
            images[n_image] = image
            labels[n_image] = label

            # Increment the number of images processed
            n_image += 1

    # Shuffle the data
    indices = np.random.permutation(n_samples)
    images = images[indices]
    labels = labels[indices]


    return images, labels
```

### **Using the Functions**

We use these functions to load the dataset for training, testing, and validation:

```python
X_train, y_train = load_dataset(root_path=train_dir, class_names=class_names)
X_valid, y_valid = load_dataset(root_path=valid_dir, class_names=class_names)
X_test, y_test = load_dataset(root_path=test_dir, class_names=class_names)
```

---

## **Model Architecture**
- Utilizes the Xception pre-trained model as the base.
- Adds custom layers including `GlobalAveragePooling2D`, `Dropout`, and `Dense` for fine-tuning.\
#### Code:
```python
from tensorflow.keras import Sequential
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

# Load the pre-trained Xception model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # Assuming 10 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

## **Testing and Evaluation the Model**

We test the trained model using the test dataset to check its performance:

```python
test_loss, test_acc = xception.evaluate(X_test, y_test)
print("Loss    : {:.4}".format(test_loss))
print("Accuracy: {:.4}%".format(test_acc*100))
```
- **Loss Function**: Categorical Crossentropy.
- **Optimizer**: Adam optimizer.
- **Metrics**: Accuracy is used to evaluate the model.
- **Callbacks**: Includes early stopping and model checkpointing.
#### Code:
```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')

# Training the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stopping, model_checkpoint]
)
```

### **Results**
- Displays training and validation accuracy and loss over epochs.
- Visualizes performance metrics using confusion matrices and classification reports.
## Dataset Structure
Expected directory structure:
```
./data/
  train/
    Ayam Goreng/
    Rendang/
    ...
  val/
    Ayam Goreng/
    Rendang/
    ...
  test/
    Ayam Goreng/
    Rendang/
    ...
```

## Usage
1. Clone the repository.
2. Place your dataset in the required directory structure (`train/`, `val/`, `test/` folders).
3. Open the notebook and execute cells sequentially.
4. Save the trained model for deployment or further inference tasks.
## Acknowledgments
- TensorFlow and Keras for model development.
- Kaggle for providing data storage and computation resources.

Feel free to modify and extend this project as needed.

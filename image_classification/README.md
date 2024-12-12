# **Machine Learning Project Documentation**

This document provides an overview of the steps and methodologies involved in building a machine learning project focused on food classification. The project includes stages from dataset preparation, model training, and evaluation.

---

## **1. Dataset**

The dataset used for this project is available from Kaggle:  
[Dataset Link](https://www.kaggle.com/datasets/rizkyyk/dataset-food-classification/data)

### **Dataset Organization**

The dataset is organized by renaming and formatting image files into their respective folders. The following steps are taken for organization:
1. **Renaming Files**: Image filenames are updated to match their respective folder names, including numbers (e.g., `cat_001.jpg`).
2. **Image Format Conversion**: Unsupported image formats are converted to `.jpg`.
3. **Removal of Original Files**: The original image files are removed after renaming to prevent duplication.

The code used for this process is located in:  
[`organize_image.py`](https://github.com/NV-Bite/Develop-ML/blob/main/image_classification/organize_image.py)

---

## **2. Dataset Splitting**

The dataset is split into three sets: `train`, `test`, and `valid`. The splitting process is done in the following notebook:  
[`data_splitting.ipynb`](https://github.com/NV-Bite/Develop-ML/blob/main/image_classification/split_dataset.ipynb)

The output of this process is three folders containing the appropriate datasets:
- `train`: Training data
- `valid`: Validation data
- `test`: Testing data

---

## **4. Dataset Preparation**

### **Loading an Image**

This function loads an image, resizes it, and normalizes its pixel values.

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

This function loads images and their corresponding labels into arrays for model training, testing, or validation.

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

To load the dataset for training, testing, and validation:

```python
X_train, y_train = load_dataset(root_path=train_dir, class_names=class_names)
X_valid, y_valid = load_dataset(root_path=valid_dir, class_names=class_names)
X_test, y_test = load_dataset(root_path=test_dir, class_names=class_names)
```

---

## **5. Model Architecture**

The model uses a pre-trained Xception model as the base and adds custom layers for fine-tuning.

#### Code for Model Architecture:
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

---

## **6. Model Testing and Evaluation**

Evaluate the model's performance on the test dataset:

```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Loss    : {:.4}".format(test_loss))
print("Accuracy: {:.4}%".format(test_acc * 100))
```

### **Callbacks**

The following callbacks are used during training:
- **EarlyStopping**: Stops training when validation loss does not improve.
- **ModelCheckpoint**: Saves the best model based on validation accuracy.

#### Code for Callbacks:
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

---

## **7. Results**

- Training and validation accuracy and loss are displayed and visualized.
- Performance metrics such as confusion matrices and classification reports are generated.

---

## **8. Usage**

1. Clone the repository.
2. Prepare your dataset and organize it into `train/`, `valid/`, and `test/` folders.
3. Open the notebook and execute the cells sequentially.
4. Save the trained model for deployment or further inference tasks.

---

## **9. Acknowledgments**

- **TensorFlow and Keras** for model development.
- **Kaggle** for providing data storage and computation resources.

Feel free to modify and extend this project as needed.

--- 

This structure ensures that all steps, methods, and tools used in the project are clearly outlined and easy to follow. Let me know if you need further adjustments!

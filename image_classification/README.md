# **Notes on Building a Machine Learning Project**

This document explains the steps and insights involved in building a machine learning project. It includes details from preparing the dataset to training and testing the model.

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

## **Testing the Model**

We test the trained model using the test dataset to check its performance:

```python
test_loss, test_acc = xception.evaluate(X_test, y_test)
print("Loss    : {:.4}".format(test_loss))
print("Accuracy: {:.4}%".format(test_acc*100))
```

---

## **Improving the Model with Hyperparameter Tuning**

### **Defining the Model**

The model is built using this function, which includes options for changing the number of layers, dropout rates, and units:

```python
def build_model(hp, n_classes=13):
    # Define all hyperparameters
    n_layers = hp.Choice('n_layers', [0, 2, 4])
    dropout_rate = hp.Choice('rate', [0.2, 0.4, 0.5, 0.7])
    n_units = hp.Choice('units', [64, 128, 256, 512])

    # Xception model with ImageNet weights, without the top classification layer
    xception_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Model architecture
    model = Sequential([
        xception_model,  # Now it's an instantiated model
        GlobalAveragePooling2D(),
    ])

    # Add hidden/top layers
    for _ in range(n_layers):
        model.add(Dense(n_units, activation='relu', kernel_initializer='he_normal'))

    # Add Dropout Layer
    model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(n_classes, activation='softmax'))

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),  # Define the learning rate
        metrics=['accuracy']
    )

    # Return the model
    return model
```

### **Random Search for the Best Settings**

We use Random Search to find the best hyperparameters:

```python
random_searcher = kt.RandomSearch(
    hypermodel=build_model,
    objective='val_loss',
    max_trials=10,
    seed=42,
    project_name="XceptionSearch",
)
random_searcher.search(train_generator, validation_data=validation_generator, epochs=10)
```

### **Callbacks for Training**

These callbacks help improve training by saving the best model and stopping early if the model stops improving:

```python
from keras.callbacks import EarlyStopping, ModelCheckpoint
```

### **Training the Best Model**

We train the best model from the Random Search:

```python
best_xception = build_model(random_searcher.get_best_hyperparameters(num_trials=1)[0])
best_xception.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(LEARNING_RATE*0.1),
    metrics=['accuracy']
)
best_xception.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    callbacks=[
        EarlyStopping(patience=2, restore_best_weights=True),
        ModelCheckpoint("BestXception.h5", save_best_only=True)
    ]
)
```

---

## **Loading the Best Model**

You can load the saved model for testing or deployment:

```python
best_xception = tf.keras.models.load_model('model.h5')
best_xception.summary()
```

---

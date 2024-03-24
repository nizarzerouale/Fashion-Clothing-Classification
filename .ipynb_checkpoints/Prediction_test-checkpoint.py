import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Function to load and prepare the image in the right format
def load_image(filename):
    # Load the image
    img = load_img(filename, color_mode='grayscale', target_size=(28, 28))
    # Convert the image to array
    img = img_to_array(img)
    # Reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # Prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

# Function to load the model and make a prediction
def predict_image_class(model, image):
    # Predict the class
    result = model.predict(image)
    # Get the index of the highest probability
    prediction = np.argmax(result, axis=1)
    return prediction[0]

# Load the image
filename = 'sample_image.png'
img = load_image(filename)

# Load the model
model = load_model('Fashion_MNIST_model.h5')

# Make a prediction
prediction = predict_image_class(model, img)

# Map the prediction to the actual class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
predicted_class = class_names[prediction]

print(f'Predicted class: {predicted_class}')

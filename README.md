**🌱 Plant Disease Detection**

A deep-learning based web application that detects plant leaf diseases using a Convolutional Neural Network (CNN).
The model is trained on the PlantVillage dataset and deployed using Streamlit for an interactive web interface.

**📌 Features**


Upload plant leaf images (JPG/PNG)

Real-time disease prediction using a trained TensorFlow model

Displays prediction label with high accuracy

Clean and simple Streamlit UI

Lightweight and easy to run locally



**🧠 Model Information**


Framework: TensorFlow / Keras

Model Type: Convolutional Neural Network (CNN)

Input Size: 128 × 128 × 3

Trained for: 38 plant diseases + healthy classes



**Trained using:**

image_dataset_from_directory()


**Model file included:**

SavedModel.h5


**📂 Project Structure**

Plant-Disease-Detection/
│

├── main.py              # Streamlit web app

├── SavedModel.h5        # Trained CNN model

├── requirements.txt     # Python dependencies

├── README.md            # Project documentation
└── .gitignore

**🚀 How to Run**

1️⃣ Clone the repository
git clone https://github.com/ROBINBHANDARI01/plant-disease-detection.git
cd plant-disease-detection

2️⃣ Install dependencies

It’s recommended to use a virtual environment:

pip install -r requirements.txt

3️⃣ Run the Streamlit app

streamlit run main.py

The app will launch in your browser.

**🖼️ How it Works**

You upload a plant leaf image.

Image is resized to 128×128 and converted to float32.

Model predicts disease class index.

Class index is mapped to the correct label derived during training.

**🗂️ Dataset**

This project uses the PlantVillage dataset.

⚠️ The dataset is not included in this repository due to size and licensing restrictions.

You can download it from Kaggle or PlantVillage:

[[https://www.kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

**🛠️ Technologies Used**

Python

TensorFlow / Keras

Streamlit

NumPy

Pillow

**📌 Requirements**

Example requirements.txt:

tensorflow

streamlit

matplotlib 

pillow

numpy


📄 License

This project is for educational & research purposes.
Dataset belongs to the original PlantVillage authors.

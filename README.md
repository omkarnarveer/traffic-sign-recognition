# 🚦 Traffic Sign Recognition System

This is a **Traffic Sign Recognition** system using **Convolutional Neural Networks (CNN)** and **OpenCV**. The system allows users to upload an image of a traffic sign, and it predicts the sign's class using a trained deep-learning model.

## 📌 Features
- ✅ **Real-time Image Upload:** Users can upload images for prediction.
- ✅ **CNN Model:** Trained with traffic sign images to recognize different classes.
- ✅ **Flask-based UI:** Built using Flask with Bootstrap 5 for an elegant and responsive interface.
- ✅ **OpenCV for Image Processing:** Preprocessing and handling image input.
- ✅ **Fast and Accurate Predictions** with a trained model.

---

## 🚀 Installation & Setup

### 🔹 1. Clone the Repository

git clone https://github.com/omkarnarveer/traffic-sign-recognition.git
cd traffic-sign-recognition
### 🔹 2. Create and Activate Virtual Environment (Optional but Recommended)
python -m venv venv

### Activate Virtual Environment:
### Windows
venv\Scripts\activate
### Mac/Linux
source venv/bin/activate
### 🔹 3. Install Dependencies
pip install -r requirements.txt

### 🔹 4. Train the Model (If not already trained)

python model/train_model.py
This will train a CNN model and save it as model/traffic_sign_model.h5.

### 🔹 5. Run the Flask Application
python app.py
### 🔹 6. Open in Browser

Once the server is running, open your browser and visit:
http://127.0.0.1:5000/
Upload an image to get the predicted traffic sign name.


## ⚙️ Technologies Used
🔹Python 3.8+
🔹Flask (Backend for handling uploads and predictions)
🔹TensorFlow/Keras (Deep learning framework for CNN)
🔹OpenCV (Image processing)
🔹Bootstrap 5 (Responsive UI)
🔹NumPy, Pandas, Scikit-learn (Data handling)

## 🤝 Contributing
Pull requests and contributions are welcome! Feel free to fork the repo and submit PRs.

## ⚠️ License
This project is open-source and free to use.


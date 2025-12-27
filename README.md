# 🇱🇰 Sinhala Character Recognition (OCR)

A Machine Learning–based **Sinhala Character Recognition System** built using **Python and K-Nearest Neighbors (KNN)**.  
This project trains a model using handwritten Sinhala characters and predicts the corresponding character from an image input.

---

## ✨ Features

✔ Build a Sinhala character dataset  
✔ Train a K-Nearest Neighbors (KNN) classifier  
✔ Save & load trained model  
✔ Predict unseen Sinhala characters  
✔ (Optional) GUI interface for testing  
✔ Simple & beginner-friendly code structure  

---

## 📂 Project Structure

```
Sinhala-Character-Recognition/
│
├── Data set creation.py
├── Traning KNN.py
├── GUI.py
├── data.npy
├── target.npy
├── sinhala-character-knn.sav
└── README.md
```

---

## 🛠️ Technologies Used

- Python
- NumPy
- OpenCV
- scikit-learn
- tkinter (for GUI)
- pickle / joblib

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/oshadha-001/Sinhala-Character-Recognition.git
cd Sinhala-Character-Recognition
```

### 2️⃣ Install Dependencies
```bash
pip install numpy opencv-python scikit-learn matplotlib
```

If using GUI:
```bash
pip install pillow
```

---

## 📸 Dataset

You can **create your own dataset** using:

```bash
python "Data set creation.py"
```

This script:
- Captures images / drawings of Sinhala characters
- Stores pixel data in `data.npy`
- Stores labels in `target.npy`

---

## 🧠 Model Training

Train the KNN model using:

```bash
python "Traning KNN.py"
```

This script will:
✔ Load dataset  
✔ Train KNN  
✔ Save model as `sinhala-character-knn.sav`  

---

## 🔍 Prediction / Testing

Run the GUI app:

```bash
python GUI.py
```

or use your own script to load the model:

```python
import pickle
model = pickle.load(open("sinhala-character-knn.sav","rb"))
```

---

## 📊 Example Workflow

1️⃣ Create dataset  
2️⃣ Train model  
3️⃣ Load saved model  
4️⃣ Predict Sinhala characters  

---

## 🏆 Future Improvements

🔹 CNN deep-learning model  
🔹 Larger dataset  
🔹 Support full Sinhala alphabet  
🔹 Mobile / Web app interface  

---

## 🤝 Contributing

Pull requests are welcome!  
If you’d like to improve accuracy or add features, feel free to fork and submit changes.

---

## 👤 Author

**Oshada Thinura**

📌 GitHub:  
https://github.com/oshadha-001

---

## 📜 License

This project is for **educational & research purposes**.

---

### ⭐ If you like this project — don’t forget to star the repo!


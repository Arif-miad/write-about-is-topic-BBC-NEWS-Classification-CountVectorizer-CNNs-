

# 📰 BBC News Classification (CountVectorizer & CNNs)
<div align="center">
      
<H2>
</H2>  
     </div>

<body>
<p align="center">
  <a href="mailto:arifmiahcse952@gmail.com"><img src="https://img.shields.io/badge/Email-arifmiah%40gmail.com-blue?style=flat-square&logo=gmail"></a>
  <a href="https://github.com/Arif-miad"><img src="https://img.shields.io/badge/GitHub-%40ArifMiah-lightgrey?style=flat-square&logo=github"></a>
  <a href="https://www.linkedin.com/in/arif-miah-8751bb217/"><img src="https://img.shields.io/badge/LinkedIn-Arif%20Miah-blue?style=flat-square&logo=linkedin"></a>

 
  
  <br>
  <img src="https://img.shields.io/badge/Phone-%2B8801998246254-green?style=flat-square&logo=whatsapp">
  
</p>

## 📖 Overview

This project showcases **BBC news article classification** using **CountVectorizer** for text feature extraction and **Convolutional Neural Networks (CNNs)** for classification. We leverage the **BBC News Dataset**, consisting of articles from categories like business, politics, sport, entertainment, and tech. By applying these techniques, we can effectively predict the category of a given news article.

---

## 📁 Dataset

The dataset used in this project is the **BBC News Dataset**. It contains **2,225 documents**, each classified into one of the following five categories:
- Business
- Entertainment
- Politics
- Sport
- Tech

Each document represents a news article, and our task is to classify them based on their content.

---

## 🛠️ Approach

### 1. **Text Preprocessing**
   - **Tokenization**: Split the text into individual words.
   - **Lowercasing**: Convert all text to lowercase for uniformity.
   - **Stop Word Removal**: Remove common stop words that do not contribute to classification.
   - **Stemming/Lemmatization**: Reduce words to their root form.

### 2. **Feature Extraction (CountVectorizer)**
   We use **CountVectorizer** to convert text documents into a numerical format (bag-of-words), where:
   - Each document is represented as a vector of word counts.
   - The vector forms the input features for the classification model.

### 3. **Model Selection (CNNs)**
   - **Convolutional Neural Networks (CNNs)** are utilized for classifying the text data.
   - The CNN model captures local patterns in the text using **1D convolution**.

---

## ⚙️ Model Architecture

We used **Keras** to implement a CNN for this task. The architecture includes:

- **Embedding Layer**: Converts words into dense vector representations.
- **1D Convolution Layer**: Detects local word patterns.
- **Pooling Layer**: Reduces the dimensionality.
- **Fully Connected Layers**: Makes predictions based on the features learned by the CNN.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense

# Define the CNN model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
```

---

## 📊 Results

The model's performance is evaluated based on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

**Sample Results**:
- **Accuracy**: 93%
- **Precision**: 92%
- **Recall**: 91%

---

## 🚀 How to Run the Project

### 1. **Install Dependencies**

First, install the required Python packages:

```bash
pip install tensorflow scikit-learn numpy pandas
```

### 2. **Clone the Repository**

```bash
git clone https://github.com/your-username/bbc-news-classification.git
cd bbc-news-classification
```

### 3. **Data Preprocessing**

Preprocess the data by running the script:

```bash
python preprocess.py
```

### 4. **Train the Model**

Run the training script:

```bash
python train.py
```

### 5. **Evaluate the Model**

Evaluate the trained model on the test set:

```bash
python evaluate.py
```

---

## 📂 Project Structure

```bash
bbc-news-classification/
├── data/
│   └── bbc_news_data.csv         # Dataset
├── models/
│   └── cnn_model.py              # CNN model implementation
├── preprocess.py                 # Preprocessing script
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
└── README.md                     # Project documentation
```

---

## 🧪 Sample Code Snippet

Here's how we apply **CountVectorizer** to transform the text into features:

```python
from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=5000)

# Fit and transform the training data
X_train = vectorizer.fit_transform(train_texts).toarray()
X_test = vectorizer.transform(test_texts).toarray()
```

---

## 🤝 Contributing

We welcome contributions! If you'd like to add improvements, new features, or fix bugs, feel free to fork this repository and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This README provides an overview of the project, guiding users through the necessary steps to understand, run, and contribute to the BBC News Classification project using CountVectorizer and CNNs.

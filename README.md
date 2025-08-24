#  Plagiarism Detection using Word2Vec + Deep Neural Network

This project is a **Plagiarism Detection System** built with **Word2Vec embeddings** and a **Deep Neural Network (DNN)**.  
The goal is to detect whether a *suspected text* is plagiarized from a *source text*.  

The app is implemented using **Streamlit** for an interactive web interface.

---

##  Features
- Input **two texts**: a source text and a suspected text.
- Converts both texts into **Word2Vec embeddings**.
- Uses a **Deep Neural Network (DNN)** trained on embeddings to classify:
  -  **No Plagiarism**
  -  **Plagiarism Detected**
- Easy-to-use **Streamlit web interface**.

---

##  Project Structure
``` bash
plagiarism_checker/
│
├── app.py # Streamlit app
├── best_model.pkl # Trained DNN model
├── word2vec_model.pkl # Trained Word2Vec embeddings
├── requirements.txt # Dependencies
├── README.md # Documentation
```


---

##  Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/plagiarism_checker.git
   cd plagiarism_checker
```


2. **Create virtual environment (recommended)**
```
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scriptsctivate      # On Windows
````

3. **Install dependencies**
```
pip install -r requirements.txt
```

4. **Download NLTK tokenizer**
```
import nltk
nltk.download('punkt')
```
## ▶️ Running the App

Run the Streamlit app:

```bash
streamlit run app.py
```

##  How it Works

###  Text Preprocessing
- Tokenization using `nltk.word_tokenize`
- Lowercasing and cleaning
- Convert each word into a **Word2Vec vector**

###  Feature Extraction
- Each text is represented as the **average Word2Vec embedding** of its words.
- Source and suspected embeddings are **concatenated** into one feature vector.

###  Model Prediction
- The concatenated embedding vector is passed to the trained **Deep Neural Network (DNN)**.
- **Output:**
  - `1` →  **Plagiarized**
  - `0` →  **Not Plagiarized**

---

##  Sample Usage

**Source Text:**
Machine learning is a branch of artificial intelligence that focuses on building systems which can learn from data and improve performance without being explicitly programmed.


**Suspected Text (Plagiarized):**
Machine learning is a part of AI that builds systems capable of learning from data and improving performance without explicit programming.


**Output:**
 Plagiarism Detected!

---

##  Model Details
- **Embeddings**: Word2Vec (`gensim`)
- **Classifier**: Deep Neural Network (`TensorFlow/Keras`)
- **Input Shape**: `(1, 1, 200)`  
  *(200 = embedding dimension × 2 (source + suspected))*
- **Output**: Binary classification  
  - `0 = Not Plagiarized`  
  - `1 = Plagiarized`

---


##  Future Improvements
- Add **Random Forest classifier** alongside DNN for comparison.
- Use **TF-IDF + cosine similarity** as a baseline.
- Extend to **code plagiarism detection** (AST + embeddings).
- Deploy app on **Streamlit Cloud **.

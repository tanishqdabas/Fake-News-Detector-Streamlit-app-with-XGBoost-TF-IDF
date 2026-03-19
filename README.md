# Fake News Detector 📰🔍

An AI-powered fake news detection web app built with **Streamlit** and **XGBoost**. Enter any news article and get instant classification — real or fake.

---

## Features

- 🤖 **ML-Powered Detection** — XGBoost classifier trained on real news data
- 📊 **TF-IDF Vectorization** — Text preprocessing with stemming & stopword removal
- 🎨 **Animated UI** — Lottie animations for an engaging experience
- ⚡ **Instant Results** — Real-time classification with a single click

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend/UI | Streamlit |
| ML Model | XGBoost (binary:logistic) |
| Text Processing | TF-IDF Vectorizer, NLTK |
| Animations | Lottie (streamlit-lottie) |
| Language | Python 3.9+ |

---

## Getting Started

### Prerequisites
- Python 3.9+

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/fake-news-detector.git
   cd fake-news-detector
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('stopwords')"
   ```

5. **Train the model** *(skip if model.pkl and vectorizer.pkl already exist)*
   ```bash
   python vectorizer_train.py   # Creates vectorizer.pkl
   python model_train.py        # Creates model.pkl
   ```

6. **Run the app**
   ```bash
   streamlit run model.py
   ```

---

## Project Structure

```
fake-news-detector/
├── model.py               # Streamlit app (main entry point)
├── model_train.py         # XGBoost model training script
├── vectorizer_train.py    # TF-IDF vectorizer training script
├── model.pkl              # Trained XGBoost model
├── vectorizer.pkl         # Fitted TF-IDF vectorizer
├── train.csv              # Training dataset
├── requirements.txt       # Python dependencies
└── *.json                 # Lottie animation files
```

---

## How It Works

1. User inputs a news article into the text box
2. Text is preprocessed: lowercased, punctuation removed, stopwords filtered, words stemmed
3. TF-IDF vectorizer transforms the text into a numerical feature vector (2000 features)
4. XGBoost model predicts: **Real** or **Fake**

---

## Dataset

The model is trained on `train.csv` which contains news articles labeled as real (0) or fake (1). Columns used: `title`, `text`, `labels`.

---

## License

This project is licensed under the [MIT License](LICENSE).

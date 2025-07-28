# Emotion Detection

This project implements a text-based emotion classifier that detects emotions such as **joy, sadness, fear, anger, love**, and **surprise** from user input. It uses a machine learning model trained on labeled sentences and integrates a simple interactive UI using `ipywidgets`

---

## Dataset

I used the [Emotions dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp?resource=download), which contains over 20,000 labeled text samples across six emotions:

- `joy`
- `sadness`
- `anger`
- `fear`
- `love`
- `surprise`

---

## How It Works

1. The dataset is split into training and test sets.
2. Text is preprocessed and vectorized using **TF-IDF**.
3. A **Logistic Regression** model is trained to classify emotions.
4. The model is tested, and predictions are evaluated with accuracy, precision, recall, F1-score, and confusion matrix.
5. A widget-based UI allows user input for live prediction.

---

## Model Evaluation

### Classification Report

| Emotion   | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| anger     | 0.92      | 0.78   | 0.84     | 465     |
| fear      | 0.83      | 0.77   | 0.80     | 420     |
| joy       | 0.84      | 0.96   | 0.89     | 1199    |
| love      | 0.86      | 0.63   | 0.72     | 302     |
| sadness   | 0.89      | 0.94   | 0.91     | 1079    |
| surprise  | 0.85      | 0.47   | 0.61     | 135     |

- **Accuracy**: `0.86`
- **Macro Average F1-Score**: `0.80`

The model performs especially well on high-frequency emotions such as **joy** and **sadness**, while emotions with fewer examples such as **surprise** and **love** show lower recall.

---

### Confusion Matrix

**How to read it:**  
- Rows represent the true emotion.
- Columns represent the predicted emotion.
- The diagonal cells show correct predictions.

#### Insights:
- **Joy** and **Sadness** are most accurately predicted.
- **Love** is often confused with **Joy**, likely due to similar positive wording.
- **Surprise** is commonly confused with **Fear** and **Joy**.
- **Anger** can sometimes be misclassified as **Sadness**, reflecting overlaps in expression.

---

## Try it Yourself

You can try live predictions by running the notebook and using the input widget:

```python
from ipywidgets import interact
interact(predict_emotion, text="I’m feeling amazing today!")

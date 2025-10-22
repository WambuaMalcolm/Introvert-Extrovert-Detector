# 🧠 Introvert–Extrovert Personality Detector

This project is a simple **machine learning web app** built with **FastAPI** that predicts whether a person is an **Introvert** or an **Extrovert** based on behavioral features such as social activity, time spent alone, and event attendance.

It uses a pre-trained **scikit-learn pipeline** (with OneHotEncoder and model) and serves predictions via a web form or API endpoint.

---

## 🚀 Features

- FastAPI backend for model inference
- HTML frontend form for easy interaction
- JSON-based prediction endpoint (`/predict`)
- Data versioning and experiment tracking with **DVC**
- Model built using **scikit-learn** and **pandas**

---

## 🧩 Tech Stack

- **Python 3.10+**
- **FastAPI**
- **Uvicorn**
- **pandas**, **scikit-learn**
- **DVC** (Data Version Control)
- **Jinja2 Templates**

---

## 📦 Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/introvert-extrovert-detector.git
   cd introvert-extrovert-detector
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate     # On Linux/Mac
   venv\Scripts\activate        # On Windows
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## DVC Setup

1. **Initialize DVC for experiment tracking:**

   ```bash
   dvc init
   ```

2. **Reproduce the full training pipeline**
   ```bash
   dvc repro
   ```

## Running the App

1. **Start FastAPI Server**

   ```bash
   uvicorn app:app --reload
   ```

2. **Open in Browser**
   - Go to: http://127.0.0.1:8000
   - Fill the form and click Predict

## 🧍‍♂️ About the Project

This model aims to detect personality type (Introvert or Extrovert) using behavioral data.
It was trained on features such as:

Time spent alone

Stage fear

Social event attendance

Going outside

Feeling drained after socializing

Friends circle size

Post frequency

## 🧩 License

MIT License © 2025 Wambua Malcolm Mwangangi

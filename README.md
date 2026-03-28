# 🌿 LeafGuard: AI Plant Disease Detection & Assistant

[![GitHub Repository](https://img.shields.io/badge/GitHub-LeafGuard--AI-green?logo=github)](https://github.com/Aastha-9/LeafGuard-AI-Plant-Disease-Detection)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-blue?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React-61DAFB?logo=react)](https://reactjs.org/)
[![TensorFlow](https://img.shields.io/badge/AI-TensorFlow-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)

**LeafGuard** is a state-of-the-art, multilingual agricultural platform designed to empower farmers with instant disease diagnosis and AI-driven expert advice. Using deep learning and Large Language Models (LLM), it provides a complete toolkit for crop health management.

---

## 🚀 Key Features

*   **📸 Instant Diagnosis**: Upload or capture photos of potato leaves for instant disease identification (Early Blight, Late Blight, or Healthy).
*   **🛡️ AI Robustness (TTA)**: Uses **Test-Time Augmentation** (5-way prediction averaging) to ensure accurate results even with low-quality field photos.
*   **💬 Multilingual AI Assistant**: An integrated chatbot powered by **Google Gemini** that provides expert advice in **English, Hindi (हिंदी), and Marathi (मराठी)**.
*   **🌱 Smart Filtering**: Local color-analysis filters reject non-leaf images automatically, saving API costs and improving speed.
*   **📰 Agricultural Blog**: Curated, multilingual articles on disease prevention and crop optimization.
*   **🎨 Glassmorphic UI**: A premium, modern web interface with camera integration and real-time feedback.

---

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, Uvicorn (Unified server architecture)
- **Frontend**: React.js, Lucide Icons, Vanilla CSS (Premium Aesthetics)
- **AI Models**: 
    - **Vision**: TensorFlow / Keras (ResNet-based CNN)
    - **Assistant**: Google Gemini 1.5/Pro (LLM) with Dynamic Model Discovery
- **DevOps**: Git LFS (Large File Storage) for model weights, Git for version control.

---

## 🏗️ Project Structure

```bash
├── api/                  # FastAPI Backend
│   ├── main.py           # Core logic & Unified Static Serving
│   ├── translations.py   # Localization dictionary
│   ├── recommendations.py # Agricultural expertise
│   └── requirements.txt  # Cloud deployment dependencies
├── frontend/             # React Frontend
│   ├── build/            # Production static assets
│   └── src/              # React components & UI logic
└── models/               # Trained CNN Model Versions (Git LFS)
```

---

## ⚡ Quick Start: Local Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Aastha-9/LeafGuard-AI-Plant-Disease-Detection.git
    cd LeafGuard-AI-Plant-Disease-Detection
    ```

2.  **Environment Variables**:
    Create a `.env` file in the `api/` folder:
    ```env
    GEMINI_API_KEY=your_google_ai_studio_key_here
    ```

3.  **Run the Unified Server**:
    ```bash
    cd api
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    *Visit: http://localhost:8000*

---

## 📡 Public Sharing (Demo)

Want to show it to friends? Run this in a separate terminal while your server is on:
```bash
npx localtunnel --port 8000
```

---

## ⚠️ Model Versioning

This project uses **Git LFS** to store model weights (housed in `/models`). Ensure you have `git-lfs` installed on your machine to pull the full 340MB weights file.

---

🌿 **LeafGuard AI** — *Empowering Agriculture with Intelligent Diagnosis.*

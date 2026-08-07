# 🌲 ForestVision - Satellite Land Cover Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📌 Overview

**ForestVision** is an AI-powered Earth Observation and Satellite Imagery Analytics platform designed to automate land cover classification. Utilizing deep learning computer vision models, the system classifies satellite image patches into **Forest** and **Non-Forest** terrains in real time. 

The application features a decoupled architecture with a high-performance **FastAPI** backend microservice for inference and an interactive **Streamlit** dark-emerald analytics dashboard.

---

## ✨ Key Features

### 🛰️ Satellite Patch Analysis
* **High-Resolution Processing:** Accepts satellite patches (Sentinel-2, Landsat, RGB patches).
* **Binary Classification:** Accurate detection of canopy cover versus urban/agricultural land.
* **Confidence Scoring:** Outputs real-time model certainty metrics and latency trackers.

### ⚡ Technical Capabilities
* **Decoupled REST API:** FastAPI backend with CORS middleware and auto-docs (`/docs`).
* **Live System Health Monitoring:** Real-time health status checks between frontend and backend.
* **1-Click Execution Engine:** Automated dual-server startup batch script (`run_app.bat`).
* **JSON Payload Inspection:** Embedded raw API output viewer for developer inspection.

---

## 🛠️ Technology Stack

| Domain | Technology |
| :--- | :--- |
| **Frontend UI** | Streamlit, Custom CSS |
| **Backend API** | Python, FastAPI, Uvicorn |
| **Deep Learning** | PyTorch / TensorFlow, Torchvision, PIL |
| **Data Handling** | NumPy, OpenCV |
| **Automation & Versioning** | Windows Batch Scripting, Git, GitHub |

---

## 📐 Architecture Overview
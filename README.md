# PolymerCode AI — Polymer Resin Classifier

An end-to-end deep learning system that identifies plastic resin codes (1–7) from images and predicts the exact chemical polymer name using Custom CNN and fine-tuned VGG-16 models.

## 🚀 Features
- **Trained Deep Learning Models:**
  - **VGG-16 (Transfer Learning):** 62.96% Test Accuracy, 61.25% Macro-F1
  - **Custom CNN:** 44.44% Test Accuracy, 37.70% Macro-F1
- **Saved Model Artifacts:** Complete models and weights/biases saved in both `.pkl` (pickle) and `.pt` (PyTorch) formats.
- **FastAPI Backend:** RESTful inference service (`/predict`, `/predict-custom-model`, `/classes`).
- **Streamlit Web UI:** Minimalist, high-performance interface with 1-click test samples and probability distribution charts.

---

## ♼ Polymer Classes & Resin Identification Codes

| Code | Symbol | Short Name | Exact Chemical Polymer Name | Typical Applications |
| :---: | :---: | :---: | :--- | :--- |
| **1** | ♳ | **PET** | Polyethylene Terephthalate | Water/soda bottles, food jars, polyester fibers |
| **2** | ♴ | **HDPE** | High-Density Polyethylene | Milk jugs, detergent bottles, piping |
| **3** | ♵ | **PVC** | Polyvinyl Chloride | Plumbing pipes, window profiles, cable sheathing |
| **4** | ♶ | **LDPE** | Low-Density Polyethylene | Plastic bags, squeeze bottles, shrink wraps |
| **5** | ♷ | **PP** | Polypropylene | Yogurt cups, bottle caps, food takeout containers |
| **6** | ♸ | **PS** | Polystyrene | Disposable cups, egg cartons, Styrofoam packaging |
| **7** | ♹ | **OTHER** | Other / Multi-Layer Resins | Eyeglass lenses, multi-layer pouches, electronics |

---

## ⚡ Quick Start

### Run the Whole Project (Frontend + Backend)
```bash
python run.py
```
* **Streamlit UI:** [http://localhost:8501](http://localhost:8501)
* **FastAPI Backend:** [http://localhost:8000](http://localhost:8000)
* **Interactive API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📁 Repository Structure
```
├── app.py                      # Streamlit prediction UI
├── api.py                      # FastAPI REST service
├── model_utils.py              # Model loading & inference engine
├── run.py                      # Unified runner script
├── run.bat                     # 1-click Windows batch launcher
├── load_saved_model.py         # Standalone model test script
├── ANN_PROJECT.ipynb           # Executed Jupyter notebook with all training & evaluation outputs
├── dataset/                    # Polymer classification dataset (train/val/test)
├── models/                     # Saved models in .pkl and .pt formats
│   ├── custom_cnn.pkl
│   ├── custom_cnn_weights_and_biases.pkl
│   ├── custom_cnn_model.pkl
│   ├── custom_cnn_best.pt
│   ├── vgg16.pkl
│   ├── vgg16_weights_and_biases.pkl
│   ├── vgg16_model.pkl
│   └── vgg16_best.pt
└── results/                    # Training histories and test comparison metrics
```

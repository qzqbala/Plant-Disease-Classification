# Plant Disease Classification — AML Final

This repository contains a PyTorch-based pipeline for plant disease classification (training, evaluation, and a Streamlit demo). Download the dataset before running anything:

- Dataset: https://www.kaggle.com/datasets/karagwaanntreasure/plant-disease-detection/data
  - Place the extracted dataset in the repository root as the `Dataset/` folder.

Files and brief explanations
- [AML_final.ipynb](AML_final.ipynb)  
  - End-to-end Jupyter notebook used for dataset filtering, transforms, training, evaluation and visualizations.  
  - Contains the data splitting code (creating [Dataset_filtered](Dataset_filtered)), model training loop and metric viz. See symbol [`AML_final.predict_image`](AML_final.ipynb) for an example inference helper.
- [app.py](app.py)  
  - Streamlit app for quick inference / demo. Loads the saved model and runs inference on user-uploaded images. References the class [`app.CustomCNN`](app.py).
- [model_arch.py](model_arch.py)  
  - Reference model architecture implementation. Provides the class [`model_arch.CustomCNN`](model_arch.py) used as canonical model definition.
- [best_model_filtered.pth](best_model_filtered.pth)  
  - Binary file: saved PyTorch model weights produced by the training in the notebook (saved with `torch.save(model.state_dict(), "best_model_filtered.pth")`). Used by [app.py](app.py) for inference.
- Dataset folders
  - [Dataset](Dataset) — raw dataset (download from Kaggle and extract here).
  - [Dataset_filtered](Dataset_filtered) — filtered top-K classes and split into `train/`, `val/`, `test/` by the notebook.

How things are connected (quick)
- The notebook [AML_final.ipynb](AML_final.ipynb) builds datasets under [Dataset_filtered](Dataset_filtered), trains a model created from [`model_arch.CustomCNN`](model_arch.py) / inline `CustomCNN` and saves weights to [best_model_filtered.pth](best_model_filtered.pth).
- The Streamlit demo [app.py](app.py) defines its own `CustomCNN` (compatible architecture) as [`app.CustomCNN`](app.py) and loads [best_model_filtered.pth](best_model_filtered.pth) to run inference.
- Use the notebook helper [`AML_final.predict_image`](AML_final.ipynb) for single-image test inference.

Quick start — setup and run
1. Download dataset from Kaggle and extract to repo root so you have:
   - Dataset/ (many class subfolders)
2. Install dependencies (see `requirements.txt` below).
3. (Optional) Open and run [AML_final.ipynb](AML_final.ipynb) to:
   - Create filtered dataset: top-K classes, split into `Dataset_filtered/train|val|test`
   - Train model and produce `best_model_filtered.pth`
4. Run the Streamlit app:
   - streamlit run app.py
   - Upload an image and get top-3 predictions.

Important implementation notes
- There are two `CustomCNN` definitions:
  - [`model_arch.CustomCNN`](model_arch.py) — canonical definition (expects input 224×224).
  - [`app.CustomCNN`](app.py) — same style but defined inline in the demo. Make sure architecture and flatten dimensions match the saved `best_model_filtered.pth` when loading weights.
- The notebook saves model state dict with `model.state_dict()`; [app.py](app.py) loads the same via `model.load_state_dict(torch.load(...))`. If you change the architecture, re-train and re-save the weights.
- `best_model_filtered.pth` is a binary PyTorch state dict — do not edit it manually.

Recommended commands
- Create venv and install:
  - python -m venv .venv
  - source .venv/bin/activate  (or `.venv\Scripts\activate` on Windows)
  - pip install -r requirements.txt
- Run notebook: jupyter lab / jupyter notebook -> open [AML_final.ipynb](AML_final.ipynb)
- Run demo: streamlit run app.py

Requirements
- Minimal tested packages listed in `requirements.txt` (below). GPU/torch variant can be chosen depending on CUDA.

License & notes
- The dataset must be downloaded separately from Kaggle (link above) and placed into the `Dataset/` folder. This repo does not include the dataset.
- Models and notebook are for educational/demonstration use.

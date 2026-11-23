# Lung Cancer Detection from CT and Histopathology Images

Two independently trained CNNs behind one Flask application: one reads **CT scan
slices**, the other reads **histopathology slides**. Both classify into
adenocarcinoma, squamous cell carcinoma, or benign tissue.

The two modalities answer different questions. A CT scan is the non-invasive
first look; histopathology is the tissue diagnosis that confirms it. Keeping
them as separate models means each trains on what it can actually see, and
either can be used alone.

---

## Results

| Modality | Test accuracy | Macro-F1 | Test size |
|---|---|---|---|
| Histopathology | **96.2%** | — | 15,000 images |
| CT scan | **73.5%** | 0.74 | 264 images |

The gap is primarily data volume rather than architecture: ~15,000 histology
images against ~800 CT slices. Histology slides are also far more texturally
distinctive, since malignant tissue architecture is visible directly in the
stain, whereas malignancy on a CT slice is a subtler density and shape cue.

**73.5% on CT is not clinically useful.** Macro-F1 0.74 against 0.79
macro-precision indicates the model over-predicts some classes and misses
others. More CT data and a pretrained backbone rather than training from scratch
are the realistic routes to improving it.

The histopathology result was measured on three well-balanced classes with a
large sample — a considerably easier setting than the CT task.

---

## Application

```
GET  /              landing page
POST /ct_scan       upload a CT slice  → {"result": "Adenocarcinoma" | "Benign" | "Squamous Cell Carcinoma"}
POST /histo_image   upload a slide     → {"result": "Lung adenocarcinoma" | "Lung benign Tissue" | "Lung squamous cell carcinoma"}
```

Flask serves three templates and loads both Keras models at startup.

---

## Layout

```
app/app.py                          Flask application, both models
app/templates/                      index, ct_scan, histo_image
app/ct_scan_model.h5                CT classifier        (25 MB)
app/models/histopathology_model.h5  Histology classifier (40 MB)
notebooks/ct-scan-cnn.ipynb         CT training and evaluation
notebooks/histopathology-cnn.ipynb  Histology training and evaluation
```

## Usage

```bash
cd app
pip install flask tensorflow pillow numpy
python app.py
```

Both trained models are committed, so the application runs without retraining.

## Datasets

Not included — ~16,000 images in total. Retraining needs:

- **CT** — three class directories under `ct scan/Data/{train,test}`:
  `adenocarcinoma`, `normal`, `squamous cell carcinoma`
- **Histopathology** — three class directories under
  `histo/rawData/{train,test}`: `lungaca`, `normal`, `squamous`
  (LC25000 lung subset)

## Stack

Python · TensorFlow/Keras · Flask · OpenCV · Pillow · NumPy

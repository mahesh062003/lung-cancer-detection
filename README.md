# Lung Cancer Detection from CT and Histopathology Images

Two independently trained CNNs behind one Flask application: one reads **CT
scan slices**, the other reads **histopathology slides**. Both classify into
adenocarcinoma, squamous cell carcinoma, or benign tissue.

The two imaging modalities answer different questions. A CT scan is the
non-invasive first look; histopathology is the tissue diagnosis that confirms
it. Keeping them as separate models rather than one combined classifier means
each is trained on what it can actually see, and either can be used alone.

---

## Results

| Modality | Test accuracy | Macro-F1 | Test size |
|---|---|---|---|
| Histopathology | **96.2%** | — | 15,000 images |
| CT scan | **73.5%** | 0.74 | 264 images |

**The 23-point gap is mostly a data problem, not a modelling one.** The
histopathology set has ~15,000 images; the CT set has ~800. Histology slides
are also far more texturally distinctive at the pixel level — malignant tissue
architecture is visible directly in the stain — whereas a malignancy on a CT
slice is a subtler density and shape cue that a small CNN trained on a few
hundred images struggles to separate from normal anatomy.

73.5% on CT is **not a clinically useful number** and this README is not going
to dress it up. Macro-F1 0.74 against 0.79 macro-precision says the model is
over-predicting some classes and missing others. The realistic routes to
improving it are more CT data and a pretrained backbone rather than training
from scratch — neither of which this project did.

The histopathology result is strong but was measured on a dataset with
3 well-balanced classes and a large sample, which is a considerably easier
setting than the CT task.

---

## Application

```
GET  /              landing page
POST /ct_scan       upload a CT slice   -> {"result": "Adenocarcinoma"|"Benign"|"Squamous Cell Carcinoma"}
POST /histo_image   upload a slide      -> {"result": "Lung adenocarcinoma"|"Lung benign Tissue"|"Lung squamous cell carcinoma"}
```

Flask serves three templates and loads both Keras models at startup.

### Known rough edge

`histo_class_labels` in `app/app.py` declares four labels including a trailing
`'none'`, while the histopathology dataset has only three classes. Index 3 is
never produced, so the extra label is dead rather than harmful — but it should
be removed, and it is noted here rather than silently left for a reader to
trip over.

---

## Layout

```
app/app.py                          Flask application, both models
app/templates/                      index, ct_scan, histo_image
app/ct_scan_model.h5                CT classifier       (25 MB)
app/models/histopathology_model.h5  Histology classifier (40 MB)
notebooks/ct-scan-cnn.ipynb         CT training and evaluation
notebooks/histopathology-cnn.ipynb  Histology training and evaluation
```

## Running it

```bash
cd app
pip install flask tensorflow pillow numpy
python app.py
```

Both trained models are committed, so the application runs without retraining.

## Datasets

Not included — ~16,000 images in total. Retraining needs:

- **CT**: three class directories under `ct scan/Data/{train,test}` —
  `adenocarcinoma`, `normal`, `squamous cell carcinoma`
- **Histopathology**: three class directories under `histo/rawData/{train,test}` —
  `lungaca`, `normal`, `squamous` (LC25000 lung subset)

## Stack

Python · TensorFlow/Keras · Flask · OpenCV · Pillow · NumPy

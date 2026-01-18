ML Assignment 3 – Submission

GitHub repository:
https://github.com/stav5/ml-assignment3

Source code:

* Single hidden layer (Chapter 11 baseline):
  https://github.com/stav5/ml-assignment3/blob/main/ch11.ipynb
* Two hidden layers (extended):
  https://github.com/stav5/ml-assignment3/blob/main/ch11_extended.ipynb
* Keras implementation:
  https://github.com/stav5/ml-assignment3/blob/main/keras_2.py

Reproducibility (Windows):

1. Create environment:
   conda create -n ml-assignment3 python=3.11 -y
   conda activate ml-assignment3
2. Install dependencies:
   pip install -r requirements.txt
3. Run Keras model:
   python keras_2.py

Outputs:

* Printed metrics: train/validation/test MSE \& accuracy + macro-AUC
* Saved figures (PNG) are written to the script directory.

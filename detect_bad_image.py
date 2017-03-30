from train_RAD_model import extract_features

import pickle
import os.path
import numpy as np

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


def main():
    RAD_dir = 'RAD'
    model_dir = 'inception_v3'
    input_dir = 'bad_image/test'
    bad_category = ['good_product', 'knife', 'gun', 'nudity']

    LE_dir = os.path.join(RAD_dir, 'LabelEncoder.p')
    clf_dir = os.path.join(RAD_dir, 'RAD_model.p')
    PCA_dir = os.path.join(RAD_dir, 'RAD_pca.p')

    # load pickle objects for the model and preprocessor
    with open(LE_dir, 'rb') as LE_f:
        LE = pickle.load(LE_f)
    with open(clf_dir, 'rb') as model_f:
        clf_xgb = pickle.load(model_f)
    with open(PCA_dir, 'rb') as pca_f:
        pca_decomposer = pickle.load(pca_f)

    feature, label = extract_features(model_dir, input_dir, bad_category)

    label = LE.transform(label)
    feature_decomposed = pca_decomposer.transform(feature)

    dtest = xgb.DMatrix(feature_decomposed, missing=feature.shape[0])

    y_pred = clf_xgb.predict(dtest)
    result = LE.inverse_transform(y_pred.astype('int'))

    return result
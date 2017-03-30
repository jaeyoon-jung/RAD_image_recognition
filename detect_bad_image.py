from train_RAD_model import create_graph

import re
import pickle
import os.path
import numpy as np

import tensorflow as tf
import xgboost as xgb


def extract_test_features(model_dir, test_dir):
    list_images = [os.path.join(test_dir, f) for f in
                   os.listdir(test_dir) if re.search('jpg|JPG|jpeg|JPEG', f)]
    # pool_3.0 returns 2048 features. Empty list to add features
    features = np.empty((len(list_images), 2048))

    create_graph(model_dir)

    print('The RAD model is running...')
    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        for ind, image in enumerate(list_images):
            if not tf.gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)
            image_data = tf.gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            features[ind, :] = np.squeeze(predictions)

    tf.reset_default_graph()

    # combine all in a single list

    return features


def main():
    RAD_dir = 'RAD'
    model_dir = 'inception_v3'
    input_dir = 'bad_image/test'

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

    feature = extract_test_features(model_dir, input_dir)
    feature_decomposed = pca_decomposer.transform(feature)

    dtest = xgb.DMatrix(feature_decomposed, missing=feature.shape[0])

    y_pred = clf_xgb.predict(dtest)
    result = LE.inverse_transform(y_pred.astype('int'))

    input_images = [f for f in
                    os.listdir(input_dir) if re.search('jpg|JPG|jpeg|JPEG', f)]

    bad_category = set(filter(lambda a: a != 'good_product', result))

    #     bad_products = dict()

    #     for itm in bad_category:
    #         bad_products[itm] = [input_images[i] for i, j in enumerate(result) if j == itm]
    print()
    if len(bad_category) > 0:
        print('The feed contains following bad images')
        for itm in bad_category:
            flagged = [input_images[i] for i, j in enumerate(result) if j == itm]
            print('  {}: {}'.format(itm, ', '.join(flagged)))
    else:
        print('No bad images were detected!')

if __name__ == "__main__":
    main()
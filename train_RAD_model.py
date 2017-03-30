# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile
import argparse
import pickle

import numpy as np
from six.moves import urllib

import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


def create_graph(model_dir):
    with tf.gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


# Extracts features right before the final layer in inception-v3's
# convolutional neural network, right before classification.
# This comes with an assumption that features used to characterize ImageNet
# data are also useful for classifying other images
def extract_features(model_dir, train_dir, bad_category):
    train_category = [(category, os.path.join(train_dir, category)) for category in
                      os.listdir(train_dir) if category in bad_category]

    all_features = []
    all_labels = []
    for category_label, images_dir in train_category:
        list_images = [os.path.join(images_dir, f) for f in
                       os.listdir(images_dir) if re.search('jpg|JPG|jpeg|JPEG', f)]
        # pool_3.0 returns 2048 features. Empty list to add features
        features = np.empty((len(list_images), 2048))
        labels = []

        create_graph(model_dir)

        with tf.Session() as sess:
            next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
            for ind, image in enumerate(list_images):
                if (ind % 100 == 0):
                    print("Processing {}th image for {}".format(ind, category_label))

                if not tf.gfile.Exists(image):
                    tf.logging.fatal('File does not exist %s', image)
                image_data = tf.gfile.FastGFile(image, 'rb').read()
                predictions = sess.run(next_to_last_tensor,
                                       {'DecodeJpeg/contents:0': image_data})
                features[ind, :] = np.squeeze(predictions)
                labels.append(category_label)

        all_features.append(features)
        all_labels.append(labels)

        tf.reset_default_graph()

    # combine all in a single list
    all_features = np.vstack(all_features)
    all_labels = np.hstack(all_labels)

    return all_features, all_labels


def main():
    #     parser = argparse.ArgumentParser('Download inception-V3 model to a specified directory \
    #                                      and train an XGBoost model with deep learned features on \
    #                                      good and bad product images.')
    #     parser.add_argument('-m', '--model_dir', help = 'model directory',
    #                         default = 'imagenet')
    #     parser.add_argument('-t', '--train_dir', help = 'training data directory',
    #                         default = None)
    #     args = parser.parse_args()

    #     #assign the parsed arguments to local variable
    #     model_dir = args.model_dir
    #     train_dir = args.train_dir
    model_dir = 'inception_v3'
    train_dir = 'bad_image/train'
    bad_category = ['good_product', 'knife', 'gun', 'nudity']
    output_directory = 'RAD'

    DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(model_dir, filename)

    # download if the model is not in the specified directory
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(model_dir)

    print('extracting features using inception-V3 model')
    feature, label = extract_features(model_dir, train_dir, bad_category)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    LE = LabelEncoder()
    LE.fit(label)
    label = LE.transform(label)
    print('saving Label Encoder to output directory')
    pickle.dump(LE, open(os.path.join(output_directory, "LabelEncoder.p"), "wb"))

    pca_decomposer = PCA(n_components=100)
    pca_decomposer.fit(feature)
    feature_decomposed = pca_decomposer.transform(feature)

    print('saving PCA preprocessor to output directory')
    pickle.dump(pca_decomposer, open(os.path.join(output_directory, "RAD_pca.p"), "wb"))

    print('training XGBoost Model')
    # prepare the dataset for xgboost
    dtrain = xgb.DMatrix(feature_decomposed, label=label)

    # parameter setting
    param = {'max_depth': 7, 'eta': 0.1, 'objective': 'multi:softmax',
             'num_class': len(LE.classes_), 'silent':1, 'subsample': 0.8, 'seed': 0}
    num_round = 300
    clf_xgb = xgb.train(param, dtrain, num_round)

    print('saving the final model')
    pickle.dump(clf_xgb, open(os.path.join(output_directory, "RAD_model.p"), "wb"))


if __name__ == "__main__":
    main()

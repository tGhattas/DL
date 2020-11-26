import tensorflow as tf
import numpy as np


def get_data():
    """
    :return: pos, neg
    """
    try:
        import google.colab
        pos_file = '/content/drive/MyDrive/Colab Notebooks/DL-HW1/DL/DL-HW1/data/pos_A0201.txt'
        neg_file = '/content/drive/MyDrive/Colab Notebooks/DL-HW1/DL/DL-HW1/data/neg_A0201.txt'
    except:
        pos_file = './data/pos_A0201.txt'
        neg_file = './data/neg_A0201.txt'
    with open(neg_file) as negf:
      neg = negf.read().split('\n')
    with open(pos_file) as posf:
      pos = posf.read().split('\n')

    return pos, neg


def onehot_vec(char):
    index = ord(char) - ord('A')
    vec = np.zeros(26, dtype=np.int)
    vec[index] = 1
    return vec


def encode_pep(peptide, one_d):
    encoded = []
    for c in peptide:
        if one_d:
            encoded.extend(onehot_vec(c))
        else:
            encoded.append(onehot_vec(c))
    return np.array(encoded)


def prepare_data(test_percent=0.1, one_d=False):
    pos, neg = get_data()
    labeled_pos_onehot_vecs = [[encode_pep(pep, one_d), 1] for pep in pos if len(pep) > 0]
    labeled_neg_onehot_vecs = [[encode_pep(pep, one_d), 0] for pep in neg if len(pep) > 0]
    labeled_onehots = np.array(labeled_pos_onehot_vecs+labeled_neg_onehot_vecs)
    np.random.shuffle(labeled_onehots)
    partition_index = int((1-test_percent)*len(labeled_onehots))
    train_data = labeled_onehots[:partition_index]
    test_data = labeled_onehots[partition_index:]
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for item in train_data:
        x_train.append(item[0])
        y_train.append(item[1])
    for item in test_data:
        x_test.append(item[0])
        y_test.append(item[1])
    return (x_train, y_train), (x_test, y_test)



if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = prepare_data()

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)

import os
import pandas as pd
import yaml
import string
import tqdm
import numpy as np
import gensim.downloader as api
from tensorflow.keras.utils import Sequence


def get_config(config_file="./config.yaml"):
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)
    return config


class Text2Vec:
    def __init__(self, config):
        self.config = config
        # Load stop words
        self.load_stop_words()
        # Load word2vec model
        self.wv = api.load("glove-wiki-gigaword-300")

    def load_stop_words(self):
        with open(self.config["dataset"]["stop_words"]) as fi:
            stop_words = fi.readlines()
            stop_words = [word.strip().lower() for word in stop_words]
            self.stop_words = stop_words

    def tokenize(self, text):
        # Remove punctuation
        translator = text.maketrans(string.punctuation, " " * len(string.punctuation))
        text = text.translate(translator)
        # Split into tokens
        tokens = text.split()
        # Remove token which is not alphabet
        tokens = [token for token in tokens if token.isalpha()]
        # Remove stop words and convert token to lowercase
        tokens = [token.lower() for token in tokens if token not in self.stop_words]
        # Remove short tokens
        tokens = [token for token in tokens if len(token) > 1]
        # Remove tokens which does not word2vec model
        tokens = [token for token in tokens if token in self.wv.index_to_key]
        return tokens

    def convert(self, text):
        tokens = self.tokenize(text)
        # Trim if need
        if len(tokens) > self.config["training"]["text_len"]:
            tokens = tokens[: self.config["training"]["text_len"]]
        vectors = [self.wv[token] for token in tokens]
        return vectors


class DataGenerator(Sequence):
    def __init__(self, config, subset):
        assert subset in ["train_set", "val_set", "test_set"]
        self.config = config
        self.subset = subset

        # Load text2vec model
        self.text2vec = Text2Vec(config)
        # Load dataset
        print(">>> Loading dataset")
        self.data_frame = pd.read_csv(open(self.config["dataset"][subset]))
        # Write labels to file
        if subset == "train_set":
            self.write_labels_to_file()
        # Read labels from file
        self.idx2label, self.label2idx = self.load_labels()

    def write_labels_to_file(self):
        label_file = os.path.join(
            self.config["training"]["log_dir"], self.config["training"]["labels"]
        )
        with open(label_file, "w") as fo:
            labels = self.data_frame[
                self.config["training"]["use_label"]
            ].values.tolist()
            labels = set(labels)
            labels = sorted(labels)
            fo.write("\n".join(labels))

    def load_labels(self):
        label_file = os.path.join(
            self.config["training"]["log_dir"], self.config["training"]["labels"]
        )
        with open(label_file) as fi:
            labels = fi.readlines()
            labels = [label.strip().lower() for label in labels]
            label2idx = {}
            for idx, label in enumerate(labels):
                label2idx[label] = idx
        return labels, label2idx

    def get_input_shape(self):
        Xs, Ys = self.__getitem__(0)
        return Xs[0].shape

    def get_num_classes(self):
        return len(self.idx2label)

    def __len__(self):
        length = int(
            self.data_frame[self.config["training"]["use_label"]].shape[0]
            / self.config["training"]["batch"]
        )
        return length

    def __getitem__(self, idx):
        Xs, Ys = [], []
        try:
            batch_size = self.config["training"]["batch"]
            use_label = self.config["training"]["use_label"]
            rows = self.data_frame[["text", use_label]][
                idx * batch_size : (idx + 1) * batch_size
            ]
            for _, value in rows.iterrows():
                # Convert text to vectors
                vectors = self.text2vec.convert(value["text"])
                # Padding if need
                len_text = self.config["training"]["text_len"]
                if len(vectors) < len_text:
                    vector_size = len(vectors[0])
                    for _ in range(len_text - len(vectors)):
                        vectors.append([0.0] * vector_size)
                Xs.append(vectors)

                # Convert label to one hot
                label = value[use_label].lower()
                one_hot = [0.0] * len(self.idx2label)
                one_hot[self.label2idx[label]] = 1.0
                Ys.append(one_hot)
        except Exception as e:
            print(e)
        return np.array(Xs), np.array(Ys)

    def on_epoch_end(self):
        self.data_frame = self.data_frame.sample(frac=1).reset_index(drop=True)


if __name__ == "__main__":
    # Statis document size
    config = get_config()
    # text2vec = Text2Vec(config)
    # data_frame = pd.read_csv(open(config['dataset']['val_set']))
    # texts = data_frame['text']
    # with open('document_lengths.csv', 'w') as fo:
    #     for i in tqdm.tqdm(range(texts.shape[0])):
    #         text = texts[i]
    #         tokens = text2vec.tokenize(text)
    #         fo.write('%d\n'%len(tokens))

    generator = DataGenerator(config, subset="val_set")
    Xs, Ys = generator.__getitem__(0)
    print(Xs)
    print(Ys)
    print(Xs[0].shape)

import sys
import json
import time
import socket
import traceback
import numpy as np
from tensorflow.keras.models import load_model

sys.path.append("..")

import utils


def start_server(args):
    # Load config
    config = utils.get_config(args.config)

    # Load model
    net = load_model(config["app"]["pretrained"], compile=False)

    # Load text2vec model
    text2vec = utils.Text2Vec(config)

    # Load labels
    label_file = config["app"]["labels"]
    with open(label_file) as fi:
        labels = fi.readlines()
        labels = [label.strip().lower() for label in labels]

    # Open server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        HOST = config["app"]["host"]
        PORT = int(config["app"]["port"])
        s.bind((HOST, PORT))
        s.listen()
        while True:
            print("Server waiting at: ", PORT)
            conn, addr = s.accept()
            print("Connected by: ", addr)
            data = conn.recv(1024)
            if not data:
                continue
            runtime = time.time()
            try:
                data = json.loads(data.decode("utf-8"))
                text = data["text"]

                # Convert text to vectors
                vectors = text2vec.convert(text)
                # Padding if need
                len_text = config["app"]["text_len"]
                if len(vectors) < len_text:
                    vector_size = len(vectors[0])
                    for _ in range(len_text - len(vectors)):
                        vectors.append([0.0] * vector_size)
                vectors = np.array(vectors)
                # Predict
                netout = net.predict(vectors[None])[0]
                label = labels[np.argmax(netout)]
                result = {"ret": True, "label": label}

            except Exception as e:
                traceback.print_exc()
                result = {"ret": False}
            runtime = time.time() - runtime
            result = json.dumps(result).encode("utf-8")
            conn.sendall(result)
            print("Sent result to ", addr)
            print("Runtime: %0.4fs" % runtime)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    args = parser.parse_args()
    start_server(args)
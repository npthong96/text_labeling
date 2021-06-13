import sys
import json
import time
import socket
import requests
import re
from bs4 import BeautifulSoup
from bs4.element import Comment
import traceback
import numpy as np
import tensorflow as tf

# from tensorflow.keras.models import load_model
# from tensorflow.keras import backend as K

# K.set_learning_phase(0)

sys.path.append("..")

import utils


def my_request_get(url, timeout=1.28, text_only=True):
    try:
        if text_only:
            r = requests.head(url, timeout=timeout)
            content_type = r.headers["content-type"]
            if "text" not in content_type:
                return ""
        r = requests.get(url, timeout=timeout)
        return r
    except:
        return ""


def text_filter(element):
    # TODO
    # Cài đặt lại như Lab02
    if element.parent.name in [
        "style",
        "title",
        "script",
        "head",
        "[document]",
        "class",
        "a",
        "li",
    ]:
        return False
    elif isinstance(element, Comment):
        """Opinion mining?"""
        return False
    elif re.match(r"[\s\r\n]+", str(element)):
        """space, return, endline"""
        return False
    return True


def start_server(args):
    # Load config
    config = utils.get_config(args.config)

    # Load model
    # net = load_model(config["app"]["pretrained"], compile=False)
    gpu_fraction = config["app"]["gpu_fraction"]
    if gpu_fraction is None:
        sess = tf.Session()
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(open(config["app"]["pretrained"], "rb").read())
    name = "text_cls"
    tf.import_graph_def(graph_def, name=name)
    input_tensor = sess.graph.get_tensor_by_name(name + "/input:0")
    output_tensor = sess.graph.get_tensor_by_name(name + "/dense/Softmax:0")

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

                texts = []
                if data["text"]:
                    texts.append(data["text"])
                elif data["html"]:
                    r = my_request_get(data["html"])
                    if r:
                        soup = BeautifulSoup(r.content, "html.parser")
                        txt = soup.findAll(text=True)
                        filtered_texts = list(filter(text_filter, txt))
                        text = ", ".join(filtered_texts)
                        texts = text.split("\n")

                outputs = []
                for text in texts:
                    # Convert text to vectors
                    vectors = text2vec.convert(text)
                    if len(vectors) == 0:
                        continue
                    # Padding if need
                    len_text = config["app"]["text_len"]
                    if len(vectors) < len_text:
                        vector_size = len(vectors[0])
                        for _ in range(len_text - len(vectors)):
                            vectors.append([0.0] * vector_size)
                    vectors = np.array(vectors)[None]
                    # Predict
                    # netout = net.predict(vectors[None])[0]
                    netout = sess.run(output_tensor, feed_dict={input_tensor: vectors})
                    label = labels[np.argmax(netout[0])]
                    outputs.append(label)
                outputs = list(set(outputs))
                result = {"ret": True, "label": outputs}

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
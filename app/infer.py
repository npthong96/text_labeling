import sys
import json
import time
import socket

sys.path.append("..")

import utils


def infer(args):
    # Load config
    config = utils.get_config(args.config)

    HOST = config["app"]["host"]
    PORT = int(config["app"]["port"])

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        args = args.__dict__
        message = json.dumps(args).encode("utf-8")
        s.sendall(message)
        result = json.loads(s.recv(1024).decode("utf-8"))
        print("Result: ", result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--text", type=str, default="", help="Input text")
    parser.add_argument("--html", type=str, default="", help="html web page")
    args = parser.parse_args()
    infer(args)
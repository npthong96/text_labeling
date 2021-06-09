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
        args[
            "text"
        ] = "Elizabeth is the longest-lived and longest-reigning British monarch, the longest-serving female head of state in world history, the world's oldest living monarch, longest-reigning current monarch, and oldest and longest-serving current head of state. Elizabeth has occasionally faced republican sentiments and press criticism of the royal family, in particular after the breakdown of her children's marriages, her annus horribilis in 1992, and the death in 1997 of her former daughter-in-law Diana, Princess of Wales. However, support for the monarchy in the United Kingdom has been and remains consistently high, as does her personal popularity."
        message = json.dumps(args).encode("utf-8")
        s.sendall(message)
        result = json.loads(s.recv(1024).decode("utf-8"))
        print("Result: ", result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    args = parser.parse_args()
    infer(args)
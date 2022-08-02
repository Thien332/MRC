import os
import argparse
from transformers import RobertaTokenizer
from syndicai import PythonPredictor

# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
sample_data = (
    "My name is Nwoke Tochukwu, I'm a Machine Learning Engineer "
    "What skillsets are <mask> to be a software Engineer."
)


def run(opt):

    # Convert image url to JSON string
    sample_json = {"text": opt.text}

    # Run a model using PythonPredictor from syndicai.py
    model = PythonPredictor([])
    response = model.predict(sample_json)

    # Print a response in the terminal
    if opt.response:
        print(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=sample_data, type=str, help='URL to a sample input data')
    parser.add_argument('--response', default=True, type=bool, help='Print a response in the terminal')
    opt = parser.parse_args()
    run(opt)

import json as sysjson
import os


def read_json(filename):
    with open(filename) as json_file:
        data = sysjson.load(json_file)
    return data


def save_json(filename, data):
    print(os.getcwd())
    print(f"Saving {filename}")
    json_string = sysjson.dumps(data, indent=2)
    with open(filename, "w") as outfile:
        outfile.write(json_string)

import fire
import json
import numpy as np

def main(first_fname, second_fname):
    first_file = open(first_fname)
    second_file = open(second_fname)
    for first_line, second_line in zip(first_file, second_file):
        first_json = json.loads(first_line)
        second_json = json.loads(second_line)

        for first_feature, second_feature in zip(first_json['features'], second_json['features']):
            assert first_feature['token'] == second_feature['token']

            for first_layer, second_layer in zip(first_feature['layers'], second_feature['layers']):
                assert first_layer['index'] == second_layer['index']

                first_vec = np.array(first_layer['values'])
                second_vec = np.array(second_layer['values'])
                # dist = np.linalg.norm(first_vec - second_vec)
                cos = 1 - np.dot(first_vec, second_vec) / (np.linalg.norm(first_vec) * np.linalg.norm(second_vec))

                print(f"{first_feature['token']} ({first_layer['index']}): {cos}")

if __name__ == '__main__':
    fire.Fire(main)

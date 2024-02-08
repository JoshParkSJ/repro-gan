"""
Script for common utility functions.
"""
import json

def write_to_json(dict_to_write, output_file):
    """
    Outputs a given dictionary as a JSON file with indents.

    Args:
        dict_to_write (dict): Input dictionary to output.
        output_file (str): File path to write the dictionary.

    Returns:
        None
    """
    with open(output_file, 'w') as file:
        json.dump(dict_to_write, file, indent=4)


def load_from_json(json_file):
    """
    Loads a JSON file as a dictionary and return it.

    Args:
        json_file (str): Input JSON file to read.

    Returns:
        dict: Dictionary loaded from the JSON file.
    """
    with open(json_file, 'r') as file:
        return json.load(file)

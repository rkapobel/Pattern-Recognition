#!/usr/bin/python
import numpy as np
import math
from SyntheticValues import ClassificationValuesGenerator
from Classificator import Classificator
from Plotter import plotClasses
import argparse

parser = argparse.ArgumentParser(description="Classificator of K classes with D = 2.")
parser.add_argument("-k", action="store", dest="numberOfDataPerClass", type=list, default=[10, 10, 10],
                    help="Number of data to generate per classes.")

if __name__ == "__main__":
    results = parser.parse_args()
    if len(results.numberOfDataPerClass) > 1:
        svg = ClassificationValuesGenerator(0, 1, 0, 1, results.numberOfDataPerClass)
        classes = svg.getSyntheticValuesForClassification()
        classificator = Classificator()
        classificator.findW(classes)
    else:
        raise ValueError("Number of classes must be greater than 1")
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import argparse

from utility import utils
from infoExtractor import InfoExtractor

parser = argparse.ArgumentParser(description="Test documents on models")

parser.add_argument(
    "path_to_resume",
    metavar="path_to_resume",
    nargs="?",
    default=None,
    help="the path to a resume",
)

parser.add_argument(
    "--profile",
    dest="model_name",
    type=str,
    default="model",
    nargs="?",
    help="the role for which the candidata has applied",
)

args = vars(parser.parse_args())
path_to_resume = args["path_to_resume"]

util = utils()
infoExtractor = InfoExtractor(util.nlp, util.parser)
util.test(path_to_resume, infoExtractor)


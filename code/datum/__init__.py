"""
PUBLIC DATuM interface.
"""
from .log import print_title
from .exchange import load 
from .preprocessor import preprocess_data
from .parser import InputFile

input_data = InputFile().data

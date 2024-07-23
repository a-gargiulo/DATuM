"""
PUBLIC DATuM Interface.
"""
from .log import print_title
from .exchange.load import load_matlab_data
from .preprocessor import preprocess_data
from .parser import InputFile

input_data = InputFile().data

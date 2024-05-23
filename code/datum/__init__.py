"""
Initialize the public interface of DATuM.
"""
from . import analysis
from . import beverli
from . import plotting
from . import transformations
from .preprocessor import preprocess_data
from .profiles import extract_profile_data
from .exchange.export import export_data_to_tecplot_binary
from .exchange.load import (
    load_matlab_data,
    load_preprocessed_data,
    load_profiles
)
from .log import print_title
from .parser import InputFile
from .piv import Piv

input_data = InputFile().data

from .torch_imports import *
from .common_imports import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

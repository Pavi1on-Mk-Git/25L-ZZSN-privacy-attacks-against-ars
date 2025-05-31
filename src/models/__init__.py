from typing import Dict

from src.models.GeneralVARWrapper import GeneralVARWrapper
from src.models.VAR import VARWrapper
from src.models.audiocraft import AudiocraftModelWrapper
from src.models.figaro import FigaroWrapper

# from src.models.MAR import MARWrapper
# from src.models.RAR import RARWrapper


gen_models: Dict[str, GeneralVARWrapper] = {
    "musicgen_small": AudiocraftModelWrapper,
    "audiogen_medium": AudiocraftModelWrapper,
    "var_16": VARWrapper,
    "var_20": VARWrapper,
    "var_24": VARWrapper,
    "var_30": VARWrapper,
    "figaro": FigaroWrapper,
    # "mar_b": MARWrapper,
    # "mar_l": MARWrapper,
    # "mar_h": MARWrapper,
    # "rar_b": RARWrapper,
    # "rar_l": RARWrapper,
    # "rar_xl": RARWrapper,
    # "rar_xxl": RARWrapper,
}

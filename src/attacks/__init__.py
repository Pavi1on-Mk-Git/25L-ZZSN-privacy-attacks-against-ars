from src.attacks.data_source import DataSource

from src.attacks.extractor import FeatureExtractor


from src.attacks.mem_info import MemInfoExtractor
from src.attacks.mem_info_mar import MemInfoMARExtractor
from src.attacks.gen_memorized import GenerateCandidates
from src.attacks.gen_memorized_audio import GenerateCandidatesAudio
from src.attacks.gen_memorized_figaro import GenerateCandidatesFigaro
from src.attacks.find_memorized import ExtractMemorized
from src.attacks.find_memorized_audio import ExtractMemorizedAudio
from src.attacks.find_memorized_figaro import ExtractMemorizedFigaro
from src.attacks.llm_mia_loss import (
    LLMMIALossExtractor,
    LLMMIALossCFGExtractor,
)
from src.attacks.llm_mia import LLMMIAExtractor
from src.attacks.llm_mia_cfg import LLMMIACFGExtractor

from src.attacks.defense import (
    DefenseExtractor,
    DefenseLossExtractor,
)


from typing import Dict


feature_extractors: Dict[str, FeatureExtractor] = {
    "mem_info": MemInfoExtractor,
    "mem_info_mar": MemInfoMARExtractor,
    "llm_mia_loss": LLMMIALossExtractor,
    "llm_mia_loss_cfg": LLMMIALossCFGExtractor,
    "llm_mia": LLMMIAExtractor,
    "gen_memorized": GenerateCandidates,
    "gen_memorized_audio": GenerateCandidatesAudio,
    "gen_memorized_figaro": GenerateCandidatesFigaro,
    "find_memorized": ExtractMemorized,
    "find_memorized_audio": ExtractMemorizedAudio,
    "find_memorized_figaro": ExtractMemorizedFigaro,
    "llm_mia_cfg": LLMMIACFGExtractor,
    "llm_mia_codebooks": LLMMIACFGExtractor,
    "defense": DefenseExtractor,
    "defense_loss": DefenseLossExtractor,
}


from src.attacks.utils import (
    load_data,
    get_datasets_clf,
    load_members_nonmembers_scores,
)

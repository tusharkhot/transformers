from typing import Dict

from modularqa.inference.dataset_readers import HotpotQAReader, DatasetReader, DropReader
from modularqa.inference.participant_qa import LMQAParticipant, MathQAParticipant, \
    ModelRouter
from modularqa.inference.participant_qgen import LMGenParticipant
from modularqa.inference.quality_checkers import QualityCheckerExample, ChainOverlapScorer, \
    LMQualityChecker

MODEL_NAME_CLASS = {
    "lmgen": LMGenParticipant,
    "lmqa": LMQAParticipant,
    "qual_example": QualityCheckerExample,
    "qual_overlap": ChainOverlapScorer,
    "mathqa": MathQAParticipant,
    "model_router": ModelRouter,
    "qual_lm": LMQualityChecker
}

READER_NAME_CLASS: Dict[str, DatasetReader] = {
    "hotpot": HotpotQAReader,
    "drop": DropReader
}

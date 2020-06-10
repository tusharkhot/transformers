from typing import Dict

from modularqa.inference.dataset_readers import HotpotQAReader, DatasetReader, DropReader
from modularqa.inference.participant_qa import LMQAParticipant, MathQAParticipant, \
    ModelRouter, BoolQAParticipant
from modularqa.inference.participant_qgen import LMGenParticipant
from modularqa.inference.quality_checkers import QualityCheckerExample, ChainOverlapScorer, \
    LMQualityChecker, LMQualityOverlapChecker, DualLMQualityChecker

MODEL_NAME_CLASS = {
    "lmgen": LMGenParticipant,
    "lmqa": LMQAParticipant,
    "qual_example": QualityCheckerExample,
    "qual_overlap": ChainOverlapScorer,
    "qual_lm": LMQualityChecker,
    "qual_dual_lm": DualLMQualityChecker,
    "qual_overlap_lm": LMQualityOverlapChecker,
    "mathqa": MathQAParticipant,
    "boolqa": BoolQAParticipant,
    "model_router": ModelRouter
}

READER_NAME_CLASS: Dict[str, DatasetReader] = {
    "hotpot": HotpotQAReader,
    "drop": DropReader
}

from typing import Dict

from modularqa.inference.dataset_readers import HotpotQAReader, DatasetReader, DropReader
from modularqa.inference.participant_qa import LMQAParticipant, MathQAParticipant, \
    BoolQAParticipant, DecompRCQA, QAEnsemble
from modularqa.inference.routers import ModelRouter, ExecutionRouter
from modularqa.inference.participant_qgen import LMGenParticipant, DecompRCGenParticipant, \
    BreakLMGenParticipant
from modularqa.inference.quality_checkers import QualityCheckerExample, ChainOverlapScorer, \
    LMQualityChecker, LMQualityOverlapChecker, DualLMQualityChecker

MODEL_NAME_CLASS = {
    "lmgen": LMGenParticipant,
    "decompgen": DecompRCGenParticipant,
    "breakgen": BreakLMGenParticipant,
    "lmqa": LMQAParticipant,
    "ensembleqa": QAEnsemble,
    "decompqa": DecompRCQA,
    "qual_example": QualityCheckerExample,
    "qual_overlap": ChainOverlapScorer,
    "qual_lm": LMQualityChecker,
    "qual_dual_lm": DualLMQualityChecker,
    "qual_overlap_lm": LMQualityOverlapChecker,
    "mathqa": MathQAParticipant,
    "boolqa": BoolQAParticipant,
    "model_router": ModelRouter,
    "operation_executer": ExecutionRouter
}

READER_NAME_CLASS: Dict[str, DatasetReader] = {
    "hotpot": HotpotQAReader,
    "drop": DropReader
}

from transformers import MODEL_FOR_QUESTION_ANSWERING_MAPPING

QUESTION_MARKER = " Q: "
COMPQ_MARKER = " QC: "
SIMPQ_MARKER = " QS: "
INTERQ_MARKER = " QI: "
EOQ_MARKER = "[EOQ]"
ANSWER_MARKER = " A: "
HINT_MARKER = " H: "
HINTS_DELIM = "; "
NOANS_MARKER = "N/A"
LIST_JOINER = " + "
TITLE_DELIM = " || "


# Model Names
SQUAD_MODEL = "squad"
IF_THEN_MODEL = "if_then"
MATH_MODEL = "math"
SQUAD_LIST_MODEL = "squad_list"


# Generator names
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

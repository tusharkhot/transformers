import re


def get_indices(question_str):
    return [int(m.group(1)) for m in re.finditer("#(\d)", question_str)]


def get_predicate_args(predicate_str):
    mat = pred_match.match(predicate_str)
    if mat is None:
        return None, None
    predicate = mat.group(1)
    pred_args = mat.group(2).split(", ")
    return predicate, pred_args


def flatten_list(input_list):
    output_list = []
    for item in input_list:
        if isinstance(item, list):
            output_list.extend(flatten_list(item))
        else:
            output_list.append(item)
    return output_list


def align_assignments(target_predicate, source_predicate, source_assignments):
    target_pred, target_args = get_predicate_args(target_predicate)
    source_pred, source_args = get_predicate_args(source_predicate)
    assert target_pred == source_pred, "Predicates should match for alignment"
    assert len(target_args) == len(source_args), "Number of arguments should match for alignment"
    target_assignment = {}
    target_assignment_map = {}
    for target_arg, source_arg in zip(target_args, source_args):
        if source_arg == "?":
            assert target_arg == "?"
            continue
        if source_arg not in source_assignments:
            raise ValueError("No assignment for {} in input assignments: {}".format(
                source_arg, source_assignments
            ))
        target_assignment[target_arg] = source_assignments[source_arg]
        target_assignment_map[target_arg] = source_arg
    return target_assignment, target_assignment_map


pred_match = re.compile("(.*)\((.*)\)")


class StepConfig:
    def __init__(self, step_json):
        self.operation = step_json["operation"]
        self.question = step_json["question"]
        self.answer = step_json["answer"]

    def to_json(self):
        return self.__dict__

    def answer_question(self, assignment):
        new_question = self.question
        for k, v in assignment:
            new_question = new_question.replace(k, v)


def valid_answer(answer):
    if isinstance(answer, list) and len(answer) == 0:
        return False
    if isinstance(answer, str) and answer == "":
        return False
    if isinstance(answer, float) and answer == 0.0:
        return False
    return True

import json
import re
from json import JSONDecodeError

from modularqa.inference.model_search import ParticipantModel

pred_match = re.compile("(.*)\((.*)\)")


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



class OperationExecuter(ParticipantModel):
    def __init__(self, remodel_file):
        if remodel_file:
            with open(remodel_file, "r") as input_fp:
                input_json = json.load(input_fp)
            self.kb_lang_groups = []
            self.qid_to_kb_lang_idx = {}
            for input_item in input_json:
                kb = input_item["kb"]
                pred_lang = input_item["pred_lang_config"]
                idx = len(self.kb_lang_groups)
                self.kb_lang_groups.append((kb, pred_lang))
                for qa_pair in input_item["qa_pairs"]:
                    qid = qa_pair["id"]
                    self.qid_to_kb_lang_idx[qid] = idx
            self.operation_regex = re.compile("\[(.*)\] <(.*)> (.*)")


    def query(self, state, debug=False):
        """The main function that interfaces with the overall search and
        model controller, and manipulates the incoming data.

        :param state: the state of controller and model flow.
        :type state: launchpadqa.question_search.model_search.SearchState
        :rtype: list
        """
        ## the data
        data = state._data
        question = data["question_seq"][-1]
        qid = data["qid"]
        (kb, pred_lang) = self.kb_lang_groups[self.qid_to_kb_lang_idx[qid]]
        model_lib = self.build_models(pred_lang, kb)
        ### run the model (as before)
        if debug: print("<OPERATION>: %s, qid=%s" % (question, qid))
        m = self.operation_regex.match(question)
        assignment = {}
        for ans_idx, ans in enumerate(data["answer_seq"]):
            assignment["#" + str(ans_idx+1)] = json.loads(ans)
        answers, facts_used = self.execute_operation(model_library=model_lib,
                                                     operation=m.group(1),
                                                     model=m.group(2),
                                                     question=m.group(3),
                                                     assignments=assignment)
        if isinstance(answers, list) and len(answers) == 0:
            return []
        # copy state
        new_state = state.copy()

        ## TODO update score?

        ## add answer
        new_state._data["answer_seq"].append(json.dumps(answers))
        new_state._data["para_seq"].append("")
        new_state._data["command_seq"].append("qa")

        ## change output
        new_state.last_output = answers
        new_state._next = "gen"

        return [new_state]

    def build_models(self, pred_lang_config, complete_kb):
        model_library = {}
        kblookup = KBLookup(kb=complete_kb)
        for model_name, configs in pred_lang_config.items():
            if model_name == "math_special":
                model = MathModel(predicate_language=configs,
                                  model_name=model_name,
                                  kblookup=kblookup)
            else:
                model = ModelExecutor(predicate_language=configs,
                                      model_name=model_name,
                                      kblookup=kblookup)
            model_library[model_name] = model
        return model_library

    def execute_select(self, operation, model, question, assignments):
        assert model in self.model_library, "Model: {} not found in model library!".format(
            model)
        indices = get_indices(question)
        for index in indices:
            idx_str = "#" + str(index)
            if idx_str not in assignments:
                raise ValueError("Can not perform project operation with input arg: {}"
                                 " No assignments yet!".format(idx_str))
            question = question.replace(idx_str, json.dumps(assignments[idx_str]))
        answers, facts_used = self.model_library[model].ask_question(question)
        if operation == "select_flat":
            return list(set(answers)), facts_used
        elif operation == "select":
            return answers, facts_used
        else:
            raise ValueError("Unknown operation: {}".format(operation))

    def execute_project(self, operation, model, question, assignments):
        # print(question, assignments)
        assert model in self.model_library, "Model: {} not found in model library!".format(
            model)
        indices = get_indices(question)
        if len(indices) > 1:
            raise NotImplementedError("Can not handle more than one answer idx for project!")
        if len(indices) == 0:
            raise ValueError("Did not find any indices to project on " + str(question))
        idx_str = "#" + str(indices[0])
        if idx_str not in assignments:
            raise ValueError("Can not perform project operation with input arg: {}"
                             " No assignments yet!".format(idx_str))
        answers = []
        facts_used = []
        for item in assignments[idx_str]:
            # print(question, idx_str, item, assignments[idx_str])
            if operation == "project_values":
                new_question = question.replace(idx_str, json.dumps(item[1]))
            elif operation == "project_keys":
                new_question = question.replace(idx_str, json.dumps(item[0]))
            else:
                new_question = question.replace(idx_str, item)
            curr_answers, curr_facts = self.model_library[model].ask_question(new_question)
            facts_used.extend(curr_facts)
            if operation == "project":
                answers.append(curr_answers)
            elif operation == "project_flat":
                answers.extend(curr_answers)
            elif operation == "project_flat_unique":
                answers.extend(curr_answers)
                answers = list(set(answers))
            elif operation == "project_zip":
                answers.append((item, curr_answers))
            elif operation == "project_keys":
                answers.append((curr_answers, item[1]))
            elif operation == "project_values":
                answers.append((item[0], curr_answers))
            else:
                raise ValueError("Unknown operation: {}".format(operation))
        return answers, facts_used

    def execute_filter(self, operation, model, question, assignments):
        assert model in self.model_library, "Model: {} not found in model library!".format(
            model)
        indices = get_indices(question)
        if len(indices) > 1:
            # check which index is mentioned in the operation
            question_indices = indices
            indices = get_indices(operation)
            if len(indices) > 1:
                raise NotImplementedError("Can not handle more than one answer idx for filter!"
                                          "Operation: {} Question: {}".format(operation, question))
            else:
                for idx in question_indices:
                    # modify question directly to include the other question indices
                    if idx not in indices:
                        idx_str = "#" + str(idx)
                        if idx_str not in assignments:
                            raise ValueError("Can not perform filter operation with input arg: {} "
                                             "No assignments yet!".format(idx_str))
                        # print(question, idx_str, assignments)
                        question = question.replace(idx_str, json.dumps(assignments[idx_str]))

        idx_str = "#" + str(indices[0])
        if idx_str not in assignments:
            raise ValueError("Can not perform filter operation with input arg: {}"
                             " No assignments yet!".format(idx_str))
        answers = []
        facts_used = []
        for item in assignments[idx_str]:
            if operation.startswith("filter_keys"):
                # item should be a tuple
                (key, value) = item
                new_question = question.replace(idx_str, json.dumps(value))
                answer, curr_facts = self.model_library[model].ask_question(new_question)
                answer = answer.lower()
                if answer == "yes" or answer == "1" or answer == "true":
                    answers.append(key)
                facts_used.extend(curr_facts)
            elif isinstance(item, list):
                # raise NotImplementedError("Assignment to {} is a list of lists: {}".format(
                #     idx_str, assignments[idx_str]
                # ))
                curr_answers, curr_facts = self.execute_operation(operation=operation, model=model,
                                                                  question=question,
                                                                  assignments={idx_str: item})
                answers.append(curr_answers)
                facts_used.extend(curr_facts)
            else:
                new_question = question.replace(idx_str, item)
                answer, curr_facts = self.model_library[model].ask_question(new_question)
                answer = answer.lower()
                if answer == "yes" or answer == "1" or answer == "true":
                    answers.append(item)
                facts_used.extend(curr_facts)
        return answers, facts_used

    def execute_flatten(self, operation, model, question, assignments):
        pred, args = get_predicate_args(question)
        answers = []
        for arg in args:
            if arg not in assignments:
                raise ValueError("Can not perform flatten operation with input arg: {}"
                                 " No assignments yet!".format(arg))
            answers.extend(flatten_list(assignments[arg]))
        return answers, []

    def execute_operation(self, operation, model, question, assignments):
        if operation.startswith("select"):
            return self.execute_select(operation, model, question, assignments)
        elif operation.startswith("project"):
            return self.execute_project(operation, model, question, assignments)
        elif operation.startswith("filter"):
            return self.execute_filter(operation, model, question, assignments)
        elif operation == "flatten":
            return self.execute_flatten(operation, model, question, assignments)
        else:
            print("Can not execute operation: {}. Returning empty list".format(operation))
            return [], []



class ModelExecutor:
    def __init__(self, predicate_language, model_name, kblookup):
        self.predicate_language = predicate_language
        self.model_name = model_name
        self.kblookup = kblookup

    def ask_question(self, input_question):
        qpred, qargs = get_predicate_args(input_question)
        if qpred is not None:
            return self.ask_question_predicate(question_predicate=input_question)
        else:
            for pred_lang in self.predicate_language:
                for question in pred_lang["questions"]:
                    question_re = re.escape(question)
                    varid_groupid = {}
                    for num in range(1, 10):
                        if "\\$" + str(num) in question_re:
                            question_re = question_re.replace("\\$" + str(num),
                                                              "(?P<G" + str(num) + ">.+)")
                            varid_groupid["$" + str(num)] = "G" + str(num)

                    # print(question_re)
                    qmatch = re.match(question_re, input_question)
                    if qmatch:
                        new_pred = pred_lang["predicate"]
                        for varid, groupid in varid_groupid.items():
                            new_pred = new_pred.replace(varid, qmatch.group(groupid))
                        return self.ask_question_predicate(new_pred)

            print("No match question found for {} "
                             "in pred_lang:\n{}".format(input_question, self.predicate_language))
            return []

    def ask_question_predicate(self, question_predicate):
        qpred, qargs = get_predicate_args(question_predicate)
        for pred_lang in self.predicate_language:
            mpred, margs = get_predicate_args(pred_lang["predicate"])
            if mpred == qpred:
                if "steps" in pred_lang:
                    kb_executer = OperationExecuter(None)
                    model_library = {"kblookup": self.kblookup}
                    source_assignments = {x: x for x in qargs}
                    curr_assignment, assignment_map = align_assignments(
                        target_predicate=pred_lang["predicate"],
                        source_predicate=question_predicate,
                        source_assignments=source_assignments
                    )

                    last_answer = None
                    for step_json in pred_lang["steps"]:
                        step = StepConfig(step_json)
                        model = "kblookup"
                        new_question = step.question
                        for k, v in curr_assignment.items():
                            if k.startswith("$"):
                                new_question = new_question.replace(k, v)
                        answers = kb_executer.execute_operation(model_library=model_library,
                                                         operation=step.operation,
                                                         model=model,
                                                         question=new_question,
                                                         assignments=curr_assignment)
                        if answers:
                            curr_assignment[step.answer] = answers
                        else:
                            print(self.kblookup.kb)
                            raise ValueError("No answer found for o:{} m:{} q:{}".format(
                                step.operation, model, new_question
                            ))
                        last_answer = step.answer
                    return curr_assignment[last_answer]
                else:
                    return self.kblookup.ask_question_predicate(question_predicate)


class KBLookup:
    def __init__(self, kb):
        self.kb = kb

    def ask_question(self, question_predicate):
        return self.ask_question_predicate(question_predicate)

    def ask_question_predicate(self, question_predicate):
        predicate, pred_args = get_predicate_args(question_predicate)
        answers = []
        # print("asking " + question_predicate)
        for fact in self.kb[predicate]:
            fact_pred, fact_args = get_predicate_args(fact)
            if len(pred_args) != len(fact_args):
                raise ValueError(
                    "Mismatch in specification args {} and fact args {}".format(
                        pred_args, fact_args
                    ))
            mismatch = False
            answer = ""
            # print(pred_args, fact_args)
            for p, f in zip(pred_args, fact_args):
                if p != "?" and p != f:
                    mismatch = True
                elif p == "?":
                    answer = f
            if not mismatch:
                answers.append(answer)
            # print(mismatch, answer, answers)
        if "?" not in pred_args:
            if len(answers) == 0:
                return "no"
            else:
                return "yes"
        else:
            return answers


class MathModel(ModelExecutor):

    def __init__(self, **kwargs):
        self.func_regex = {
            "is_greater\((.+) \| (.+)\)": MathModel.greater_than,
            "is_smaller\((.+) \| (.+)\)": MathModel.smaller_than,
            "diff\((.+) \| (.+)\)": MathModel.diff,
            "belongs_to\((.+) \| (.+)\)": MathModel.belongs_to,
            "max\((.+)\)": MathModel.max,
            "min\((.+)\)": MathModel.min,
            "count\((.+)\)": MathModel.count

        }
        super(MathModel, self).__init__(**kwargs)

    @staticmethod
    def get_number(num):
        try:
            item = json.loads(num)
        except JSONDecodeError:
            print("Could not JSON parse: " + num)
            raise
        if isinstance(item, list):
            if (len(item)) != 1:
                raise ValueError("List of values instead of single number in {}".format(num))
            item = item[0]
        try:
            return float(item)
        except ValueError:
            print("Could not parse float from: " + item)
            raise

    @staticmethod
    def max(groups):
        if len(groups) != 1:
            raise ValueError("Incorrect regex for max. "
                             "Did not find 1 group: {}".format(groups))
        try:
            entity = json.loads(groups[0])
            if isinstance(entity, list):
                numbers = [MathModel.get_number(x) for x in entity]
            else:
                print("max can only handle list of entities. Arg: " + str(entity))
                return "", []
        except JSONDecodeError:
            print("Could not parse: {}".format(groups[0]))
            raise
        return max(numbers), []

    @staticmethod
    def min(groups):
        if len(groups) != 1:
            raise ValueError("Incorrect regex for min. "
                             "Did not find 1 group: {}".format(groups))
        try:
            entity = json.loads(groups[0])
            if isinstance(entity, list):
                numbers = [MathModel.get_number(x) for x in entity]
            else:
                print("max can only handle list of entities. Arg: " + str(entity))
                return "", []
        except JSONDecodeError:
            print("Could not parse: {}".format(groups[0]))
            raise
        return min(numbers), []

    @staticmethod
    def count(groups):
        if len(groups) != 1:
            raise ValueError("Incorrect regex for max. "
                             "Did not find 1 group: {}".format(groups))
        try:
            entity = json.loads(groups[0])
            if isinstance(entity, list):
                return len(entity), []
            else:
                print("max can only handle list of entities. Arg: " + str(entity))
                return "", []
        except JSONDecodeError:
            print("Could not parse: {}".format(groups[0]))
            raise

    @staticmethod
    def belongs_to(groups):
        if len(groups) != 2:
            raise ValueError("Incorrect regex for belongs_to. "
                             "Did not find 2 groups: {}".format(groups))
        try:
            entity = json.loads(groups[0])
            if isinstance(entity, list):
                if len(entity) > 1:
                    print(
                        "belongs_to can only handle single entity as 1st arg. Args:" + str(groups))
                    return "", []
                else:
                    entity = entity[0]
        except JSONDecodeError:
            entity = groups[0]
        try:
            ent_list = json.loads(groups[1])
        except JSONDecodeError:
            print("Could not JSON parse: " + groups[1])
            raise

        if not isinstance(ent_list, list):
            print("belongs_to can only handle lists as 2nd arg. Args:" + str(groups))
            return "", []
        if entity in ent_list:
            return "yes", []
        else:
            return "no", []

    @staticmethod
    def diff(groups):
        if len(groups) != 2:
            raise ValueError("Incorrect regex for diff. "
                             "Did not find 2 groups: {}".format(groups))
        num1 = MathModel.get_number(groups[0])
        num2 = MathModel.get_number(groups[1])
        if num2 > num1:
            return round(num2 - num1, 3), []
        else:
            return round(num1 - num2, 3), []

    @staticmethod
    def greater_than(groups):
        if len(groups) != 2:
            raise ValueError("Incorrect regex for greater_than. "
                             "Did not find 2 groups: {}".format(groups))
        num1 = MathModel.get_number(groups[0])
        num2 = MathModel.get_number(groups[1])
        if num1 > num2:
            return "yes", []
        else:
            return "no", []

    @staticmethod
    def smaller_than(groups):
        if len(groups) != 2:
            raise ValueError("Incorrect regex for smaller_tha. "
                             "Did not find 2 groups: {}".format(groups))
        num1 = MathModel.get_number(groups[0])
        num2 = MathModel.get_number(groups[1])
        if num1 < num2:
            return "yes", []
        else:
            return "no", []

    def ask_question(self, question):
        return self.ask_question_predicate(question)

    def ask_question_predicate(self, question_predicate):
        for regex, func in self.func_regex.items():
            m = re.match(regex, question_predicate)
            if m:
                return func(m.groups())
        raise ValueError("Could not parse: {}".format(question_predicate))


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
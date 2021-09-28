import json
import re
from json import JSONDecodeError

from modularqa.inference.executer_utils import get_indices, get_predicate_args, flatten_list, \
    align_assignments, StepConfig, valid_answer


class OperationExecuter:

    def __init__(self, model_library):
        self.model_library = model_library

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
            answers, facts_used = None, None
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
                        answers, facts_used = self.ask_question_predicate(new_pred)
                        if valid_answer(answers):
                            # if this is valid answer, return it
                            return answers, facts_used

            if answers is not None:
                # some match found for the question but no valid answer.
                # Return the last matching answer.
                return answers, facts_used
            else:
                raise ValueError("No match question found for {} "
                                 "in pred_lang:\n{}".format(input_question,
                                                            self.predicate_language))

    def ask_question_predicate(self, question_predicate):
        qpred, qargs = get_predicate_args(question_predicate)
        facts_used = []
        for pred_lang in self.predicate_language:
            mpred, margs = get_predicate_args(pred_lang["predicate"])
            if mpred == qpred:
                if "steps" in pred_lang:
                    model_library = {"kblookup": self.kblookup}
                    kb_executor = OperationExecuter(model_library)
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
                        answers, facts = kb_executor.execute_operation(operation=step.operation,
                                                                       model=model,
                                                                       question=new_question,
                                                                       assignments=curr_assignment)
                        if valid_answer(answers):
                            curr_assignment[step.answer] = answers
                            facts_used.extend(facts)
                        else:
                            return [], []
                            # print(self.kblookup.kb)
                            # raise ValueError("No answer found for o:{} m:{} q:{}".format(
                            #     step.operation, model, new_question))
                        last_answer = step.answer
                    return curr_assignment[last_answer], facts_used
                else:
                    return self.kblookup.ask_question_predicate(question_predicate)


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


class KBLookup:
    def __init__(self, kb):
        self.kb = kb

    def ask_question(self, question_predicate):
        return self.ask_question_predicate(question_predicate)

    def ask_question_predicate(self, question_predicate):
        predicate, pred_args = get_predicate_args(question_predicate)
        answers = []
        facts_used = []
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
                if p != "?" and p != f and p != "_":
                    mismatch = True
                elif p == "?":
                    answer = f
            if not mismatch:
                answers.append(answer)
                facts_used.append(fact)
            # print(mismatch, answer, answers)
        if "?" not in pred_args:
            if len(answers) == 0:
                return "no", facts_used
            else:
                return "yes", facts_used
        else:
            return answers, facts_used



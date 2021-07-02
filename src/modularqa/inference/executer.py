import json
import re
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
        answers = self.execute_operation(model_library=model_lib,
                                         operation=m.group(1),
                                         model=m.group(2),
                                         question=m.group(3),
                                         assignments=assignment)
        if len(answers) == 0:
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
            model = ModelExecutor(predicate_language=configs,
                                  model_name=model_name,
                                  kblookup=kblookup)
            model_library[model_name] = model
        return model_library

    def execute_operation(self, model_library, operation, model, question, assignments):
        if operation == "select":
            assert model in model_library, "Model: {} not found in model library: {}".format(
                model, model_library.keys())
            return model_library[model].ask_question(question)
        elif operation == "project" or operation == "project_flat" \
                or operation == "project_flat_unique":
            # print(question, assignments)
            assert model in model_library, "Model: {} not found in model library!".format(
                model)
            indices = get_indices(question)
            if len(indices) > 1:
                raise NotImplementedError("Can not handle more than one answer idx for project!")
            idx_str = "#" + str(indices[0])
            if idx_str not in assignments:
                raise ValueError("Can not perform project operation with input arg: {}"
                                 " No assignments yet!".format(idx_str))
            answers = []
            for item in assignments[idx_str]:
                new_question = question.replace(idx_str, item)
                if operation == "project":
                    answers.append(model_library[model].ask_question(new_question))
                elif operation == "project_flat":
                    answers.extend(model_library[model].ask_question(new_question))
                elif operation == "project_flat_unique":
                    answers.extend(model_library[model].ask_question(new_question))
                    answers = list(set(answers))
            return answers
        elif operation == "filter":
            assert model in model_library, "Model: {} not found in model library!".format(
                model)
            indices = get_indices(question)
            if len(indices) > 1:
                raise NotImplementedError("Can not handle more than one answer idx for filter!")
            idx_str = "#" + str(indices[0])
            if idx_str not in assignments:
                raise ValueError("Can not perform filter operation with input arg: {}"
                                 " No assignments yet!".format(idx_str))
            answers = []
            for item in assignments[idx_str]:
                if isinstance(item, list):
                    # raise NotImplementedError("Assignment to {} is a list of lists: {}".format(
                    #     idx_str, assignments[idx_str]
                    # ))
                    answers.append(self.execute_operation(model_library=model_library,
                                                          operation=operation, model=model,
                                                          question=question,
                                                          assignments={idx_str: item}))
                else:
                    new_question = question.replace(idx_str, item)
                    answer = self.model_library[model].ask_question(new_question).lower()
                    if answer == "yes" or answer == "1" or answer == "true":
                        answers.append(item)
            return answers
        elif operation == "flatten":
            pred, args = get_predicate_args(question)
            answers = []
            for arg in args:
                if arg not in assignments:
                    raise ValueError("Can not perform flatten operation with input arg: {}"
                                     " No assignments yet!".format(arg))
                answers.extend(flatten_list(assignments[arg]))
            return answers


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
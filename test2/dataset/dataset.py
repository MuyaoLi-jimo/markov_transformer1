# The script is used to generate datasets.

import numpy as np
np.random.seed(42)
import random
random.seed(42)
from prompt import Prompt
from objects import Objects
from hyperbaton import Hyperbaton
from word_sorting import WORDS
from copy import deepcopy
from message_generate import create_user_message,create_assistant_message
from utils import utils 


dataset_names = [
"coin_flip", "last_letter_concat", "reverse_list", "boolean_expressions", "dyck_languages", "multi-step_arithmetic", "navigate", "temporal_sequences", "tracking_shuffled_objects", "word_sorting", "scan", "date_understanding", "object_counting", "penguins_in_a_table", "hyperbaton",
]

with open("names.txt", "r") as f:
    NAMES = f.read().strip().split()

class Dataset_Generator:
    def __init__(self, dataset_name: str) -> None:
        self.name = dataset_name
        if dataset_name == "object_counting":
            OBJECTS = Objects()
            self.classes = OBJECTS.classes
        if dataset_name == "hyperbaton":
            HYPERBATON = Hyperbaton()
            self.classes = HYPERBATON.classes

    def generate_braket(self, pair_num):
        '''
        return incomplete braket string with #pair_num bra and its completion
        '''
        bra_ket = {"(": ")", "{": "}", "[": "]", "<": ">"}
        result = []
        stack = []
        bra_to_use = random.choices(list(bra_ket.keys()), k=pair_num)
        idx = 0
        complete_idx = []
        while bra_to_use or stack:
            if stack and bra_to_use:
                if random.random() > 0.3:
                    bra_use = bra_to_use.pop()
                    result.append(bra_use)
                    stack.append(bra_use)
                else:
                    result.append(bra_ket[stack.pop()])
            elif bra_to_use:
                bra_use = bra_to_use.pop()
                result.append(bra_use)
                stack.append(bra_use)
            else:
                result.append(bra_ket[stack.pop()])
                complete_idx.append(idx)
            idx += 1
        # choose where to split the string
        split_idx = random.choice(complete_idx)
        question_braket = result[:split_idx]
        completion_braket = result[split_idx:]

        return result, question_braket, completion_braket, complete_idx


    def word_sorting(self, words):
        n = len(words)
        # Main Loop
        for i in range(n):
            for j in range(0, n-i-1):
                # Compare Words Letter By Letter
                same = True
                for k in range(min(len(words[j]), len(words[j+1]))):
                    if words[j][k] > words[j+1][k]:
                        # Swap the words
                        words[j], words[j+1] = words[j+1], words[j]
                        same = False
                        break  # Stop comparing if a swap is made
                    elif words[j][k] < words[j+1][k]:
                        same = False
                        break  # Stop comparing if no swap is needed
                # Check If Prefix
                if same and len(words[j]) > len(words[j+1]):
                    words[j], words[j+1] = words[j+1], words[j]
        return words

    def gen_data_from_len(self, length: int) -> dict:
        '''
        return datapoint of given length,
        datapoint is a dict of keys including `"question", "gt", ...`
        '''
        if self.name == "coin_flip":
            # example question: A coin is heads up. Ka flips the coin. Sherrie flips the coin. Is the coin still heads up?
            # example answer: Yes
            names = random.sample(NAMES, length)
            flips = random.choices([True, False], k=length)
            question = "A coin is heads up. "
            for idx in range(length):
                question += f"{names[idx]} {'flips' if flips[idx] else 'does not flip'} the coin. "
            question += "Is the coin still heads up?"
            gt = "Yes" if sum(flips) % 2 == 0 else "No"
            return {"question": question,
                    "gt": gt,
                    "names": names,
                    "flips": flips}
        elif self.name == "last_letter_concat":
            # example question: Take the last letters of the words in "Velazquez, Mullins, Moreno" and concatenate them.
            # example answer: zso
            names = random.sample(NAMES, length)
            question = f"Take the last letters of the words in \"{', '.join(names)}\" and concatenate them."
            gt = ''.join([name[-1] for name in names])
            return {"question": question,
                    "gt": gt,
                    "names": names}
        elif self.name == "reverse_list":
            # example question: Reverse the sequence "Jennings, Perez, Melendez".
            # example answer: ['Melendez', 'Perez', 'Jennings']
            names = random.sample(NAMES, length)
            question = f"Reverse the sequence \"{', '.join(names)}\"."
            gt = names[::-1]
            return {"question": question,
                    "gt": gt,
                    "names": names}
        elif self.name == "boolean_expressions":
            pass
        elif self.name == "dyck_languages":
            # example question: Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: [ [
            # example answer: ] ]
            _, question_braket, complete_braket, _ = self.generate_braket(length)
            question = "Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: " + ' '.join(question_braket)
            gt = ' '.join(complete_braket)
            return {"question": question,
                    "gt": gt,
                    "question_braket": question_braket,
                    "complete_braket": complete_braket}

        elif self.name == "multi-step_arithmetic":
            pass
        elif self.name == "navigate":
            # example question: If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step backward. Take 9 steps left. Take 2 steps backward. Take 6 steps forward. Take 4 steps forward. Take 4 steps backward. Take 3 steps right.
            # example answer: No
            def calc_move(moves):
                loc = [0, 0]
                question = "If you follow these instructions, do you return to the starting point? Always face forward. "
                for move in moves:
                    if move[0] == "left":
                        loc[0] -= move[1]
                    elif move[0] == "right":
                        loc[0] += move[1]
                    elif move[0] == "forward":
                        loc[1] += move[1]
                    elif move[0] == "backward":
                        loc[1] -= move[1]
                    question += f"Take {move[1]} steps {move[0]}. "
                left_moves = [("right", -loc[0]) if loc[0] < 0 else ("left", loc[0]), ("forward", -loc[1]) if loc[1] < 0 else ("backward", loc[1])]
                return loc, left_moves, question.strip()

            moves = []
            if length >= 3:
                # randomly move until there are 2 moves left
                for _ in range(length - 2):
                    direction = random.choice(["left", "right", "backward", "forward"])
                    step = random.randint(1, 9)
                    moves.append((direction, step))
                    _, left_moves, _ = calc_move(moves)
                # decide whether go back to the initial spot
                if random.random() > 0.5 and left_moves[0][1] != 0 and left_moves[1][1] != 0:
                    moves += left_moves
                else:
                    for _ in range(2):
                        direction = random.choice(["left", "right", "backward", "forward"])
                        step = random.randint(1, 9)
                        moves.append((direction, step))
            else:
                for _ in range(length):
                    direction = random.choice(["left", "right", "backward", "forward"])
                    step = random.randint(1, 9)
                    moves.append((direction, step))
            final_loc, _, question = calc_move(moves)
            if final_loc == [0, 0]:
                gt = "Yes"
            else:
                gt = "No"

            return {"question": question,
                    "gt": gt,
                    "moves": moves}
        elif self.name == "word_sorting":
            # example question: Sort the following words alphabetically: syndrome therefrom
            # example answer: syndrome therefrom
            words = random.sample(WORDS, k=length)
            question = "Sort the following words alphabetically: " + " ".join(words)
            gt = ' '.join(self.word_sorting(words))
            return {"question": question,
                    "gt": gt,
                    "words": words}
        elif self.name == "object_counting":
            def f(x, target):
                x["label"] = target
                x["num"] = random.randint(1, 3)
                return x

            def num_to_term(obj):
                num = obj["num"]
                if num == 1:
                    term = f"{obj['sing']}"
                elif num == 2:
                    term = f"two {obj['plural']}"
                elif num == 3:
                    term = f"three {obj['plural']}"
                return term

            def calc_target(objs):
                c = 0
                for obj in objs:
                    if obj["label"]:
                        c += obj["num"]
                return c

            classes = self.classes
            # each question include objects from 1 target class & 1 other class
            target_class, other_class = random.sample(list(classes.keys()), 2)
            # decide the num of target objects
            target_object_num = random.randint(1, length)
            other_object_num = length - target_object_num
            # choose target objects and other objects
            target_objects = random.sample(classes[target_class], target_object_num)
            other_objects = random.sample(classes[other_class], other_object_num)
            target_objects = list(map(lambda x: f(x, True), target_objects))
            other_objects = list(map(lambda x: f(x, False), other_objects))
            # decide num of each objects
            objects = target_objects + other_objects
            random.shuffle(objects)
            # objects_num = [random.randint(1, 3) for _ in range(length)]
            question = "I have "
            for idx in range(length-1):
                question += num_to_term(objects[idx]) + ", "
            question += f"and {num_to_term(objects[-1])}. How many {target_class} do I have?"
            gt = calc_target(objects)
            return {"question": question,
                    "gt": gt,
                    "objects": objects,
                    "tg_class": target_class}
        elif self.name == "hyperbaton":
            # example question:
            # Which sentence has the correct adjective order:
            # Options:
            # (A) midsize old grey Brazilian sweater
            # (B) midsize grey Brazilian old sweater
            #
            # example answer
            # (A)

            # choose adj types
            assert length != 1
            adj_types = list(self.classes.keys())
            noun = random.choice(self.classes[("noun", -1)])
            adj_types.remove(("noun", -1))
            use_adj_types = sorted(random.sample(adj_types, k=length), key=lambda adj_type: adj_type[1])
            # map to corresponding adjs + noun
            use_adjs = [random.choice(self.classes[adj_type]) for adj_type in use_adj_types]
            # shuffle correct adjs to get incorrect term
            adj_type_pack = list(zip(use_adj_types, use_adjs))
            incorrect_adj_type_pack = deepcopy(adj_type_pack)
            random.shuffle(incorrect_adj_type_pack)
            while adj_type_pack == incorrect_adj_type_pack:
                random.shuffle(incorrect_adj_type_pack)
            incorrect_adj_types, incorrect_adjs = zip(*incorrect_adj_type_pack)
            incorrect_adj_types = list(incorrect_adj_types)
            incorrect_adjs = list(incorrect_adjs)

            correct_term = " ".join(use_adjs + [noun])
            incorrect_term = " ".join(incorrect_adjs + [noun])
            if random.random() > 0.5:
                term_a = correct_term
                term_b = incorrect_term
                gt = "(A)"
                a_adjs = use_adjs
                b_adjs = incorrect_adjs
                a_adj_types = use_adj_types
            else:
                term_a = incorrect_term
                term_b = correct_term
                gt = "(B)"
                a_adjs = incorrect_adjs
                b_adjs = use_adjs
                a_adj_types = incorrect_adj_types
            # question = f"Which sentence has the correct adjective order:\nOptions:\n(A) {term_a}\n(B) {term_b}"
            # return {"question": question,
            #         "gt": gt,
            #         "a_adjs": a_adjs,
            #         "b_adjs": b_adjs}
            question = f"Does the sentence has the correct adjective order: {term_a}"
            return {"question": question,
                    "gt": True if gt == "(A)" else False,
                    "a_adjs": a_adjs,
                    "a_adj_types": a_adj_types}
        else:
            raise ValueError(f"dataset {self.name} is not supported")


    def direct_IO(self, data: dict) -> dict:
        '''
        return direct answer input-output of given data
        '''
        pass

    def cot_IO(self, data: dict) -> dict:
        '''
        return CoT input-output of given data
        '''
        pass

    def rfft_IO(self, data: dict) -> dict:
        '''
        return rfft input-output of given data
        '''
        instruction = "Follow the given rule to solve the question.\nrule:"
        if self.name == "coin_flip":
            P = Prompt("coin_flip")
            rule = P.rule
            input = instruction + rule + "\n\nQ: " + data["question"]
            # rfft output
            flips = data["flips"]
            heads_up = True
            output = P.initialize.format(flips)
            for flip in flips:
                if flip:
                    output += P.one_iteration_2_1_flip.format(heads_up, not heads_up)
                    heads_up = not heads_up
                else:
                    output += P.one_iteration_2_1_no_flip.format(heads_up)
            output += P.return_result.format(heads_up, "Yes" if heads_up else "No")
            return {"input": input,
                    "output": output}
        elif self.name == "last_letter_concat":
            P = Prompt("last_letter_concat")
            rule = P.rule
            input = instruction + rule + "\n\nQ: " + data["question"]
            # rfft output
            names = data["names"]
            output = P.initialize.format(names)
            result = ""
            for name in names:
                output += P.one_iteration_2_1.format(name, result, name[-1], name[-1], result + name[-1])
                result += name[-1]
            output += P.return_result.format(result)
            return {"input": input,
                    "output": output}
        elif self.name == "reverse_list":
            P = Prompt("reverse_list")
            rule = P.rule
            input = instruction + rule + "\n\nQ: " + data["question"]
            # rfft output
            names = data["names"]
            output = P.initialize.format(names)
            result = []
            while names:
                output += P.one_iteration_2_1.format(names, result, names[-1], result + [names[-1]])
                result.append(names.pop())
            output += P.return_result.format(result,', '.join(result))

            return {"input": input,
                    "output": output}
        elif self.name == "object_counting":
            P = Prompt("object_counting")
            rule = P.rule
            input = instruction + rule + "\n\nQ: " + data["question"]
            # rfft output
            tg_class = data["tg_class"]
            objects = data["objects"]
            objects_raw = [object["plural"] for object in objects]
            output = P.initialize.format(f"[{', '.join(objects_raw)}]", tg_class)
            count = 0
            for object in objects:
                if object["label"]:
                    output += P.one_iteration_2_1_class.format(object["plural"], tg_class, count, object["num"], count + object["num"])
                    count += object["num"]
                else:
                    output += P.one_iteration_2_1_not_class.format(object["plural"], tg_class, count)
            output += P.return_result.format(count)
            return {"input": input,
                    "output": output}
        elif self.name == "dyck_languages":
            P = Prompt("dyck_languages")
            rule = P.rule
            input = instruction + rule + "\n\nQ: " + data["question"]
            # rfft output
            question_braket = data["question_braket"]
            ketbra_map = {")": "(", "}": "{", "]": "[", ">": "<"}
            s = "".join(question_braket)

            output = P.initialize_1.format(s) + P.initialize_2
            stack = []
            for braket in s:
                if braket in ketbra_map:
                    output += P.one_iteration_2_1_map.format(braket, stack, stack[-1], stack[:-1], ketbra_map[braket])
                    top = stack.pop() if stack else None
                    if ketbra_map[braket] != top:
                        raise ValueError("invalid input")
                else:
                    output += P.one_iteration_2_1_not_map.format(braket, stack, stack + [braket])
                    stack.append(braket)
            braket_map = {"(": ")", "{": "}", "[": "]", "<": ">"}
            complete = ""
            # Completion Loop
            while stack:
                output += P.one_iteration_4_1_stack.format(stack, stack[-1], braket_map[stack[-1]], complete, complete + braket_map[stack[-1]], stack[:-1])
                complete += braket_map[stack.pop()]

            output += P.one_iteration_4_1_not_stack + P.return_result.format(complete)
            assert complete == ''.join(data["complete_braket"])
            return {"input": input,
                    "output": output}
        elif self.name == "hyperbaton":
            P = Prompt("hyperbaton")
            rule = P.rule
            input = instruction + rule + "\n\nQ: " + data["question"]
            # rfft output
            adjs = data["a_adjs"]
            adj_types = data["a_adj_types"]
            adj_map = {'opinion': 0,
                       'size': 1,
                    #    'physical quality': 2,
                       'shape': 3,
                       'age': 4,
                       'color': 5,
                       'origin': 6, 'material': 7,
                    #    'type': 8,
                       'purpose': 9}

            output = P.initialize
            current_type = None
            correct = True
            for idx, adj in enumerate(adjs):
                adj_type = adj_types[idx][0]
                if current_type is None or adj_map[adj_type] > adj_map[current_type]:
                    output += P.one_iteration_2_1_later.format(adj, current_type, adj_type)
                    current_type = adj_type
                else:
                    output += P.one_iteration_2_1_earlier.format(adj, current_type, adj_type)
                    correct = False
            output += P.return_result.format(correct, "Yes" if correct else "No")
            return {"input": input,
                    "output": output}
        elif self.name == "navigate":
            P = Prompt("navigate")
            rule = P.rule
            input = instruction + rule + "\n\nQ: " + data["question"]
            # rfft output
            moves = data["moves"]
            loc = [0, 0]
            output = P.initialize.format(moves)
            # Main Loop
            for move in moves:
                if move[0] == "left":
                    output += P.one_iteration_2_1_left.format(move, loc, move[1], loc[0], loc[0] - move[1], (loc[0] - move[1], loc[1]))
                    loc[0] -= move[1]
                elif move[0] == "right":
                    output += P.one_iteration_2_1_right.format(move, loc, move[1], loc[0], loc[0] + move[1], (loc[0] + move[1], loc[1]))
                    loc[0] += move[1]
                elif move[0] == "forward":
                    output += P.one_iteration_2_1_forward.format(move, loc, move[1], loc[1], loc[1] + move[1], (loc[0], loc[1] + move[1]))
                    loc[1] += move[1]
                elif move[0] == "backward":
                    output += P.one_iteration_2_1_backward.format(move, loc, move[1], loc[1], loc[1] - move[1], (loc[0], loc[1] - move[1]))
                    loc[1] -= move[1]
            assert data["gt"] == "Yes" if loc == [0, 0] else data["gt"] == "No"
            output += P.return_result.format(loc, loc == [0, 0], data["gt"])
            return {"input": input,
                    "output": output}
        elif self.name == "word_sorting":
            P = Prompt("word_sorting")
            rule = P.rule
            input = instruction + rule + "\n\nQ: " + data["question"]
            # rfft output
            words = data["words"]
            n = len(words)
            output = P.initialize.format(words, n)
            # Main Loop
            for i in range(n):
                output += P.one_iteration_1_1_i.format(i)
                for j in range(0, n-i-1):
                    # Compare Words Letter By Letter
                    same = True
                    output += P.one_iteration_1_1_1_j.format(j, words[j], words[j+1])
                    for k in range(min(len(words[j]), len(words[j+1]))):
                        if words[j][k] > words[j+1][k]:
                            words_ = deepcopy(words)
                            words_[j], words_[j+1] = words_[j+1], words_[j]
                            output += P.one_iteration_1_1_1_1_1_k_big.format(len(words[j]), len(words[j+1]), min(len(words[j]), len(words[j+1])), k, words[j][k], words[j+1][k], words, words[j], words[j+1], words_)
                            # Swap the words
                            words[j], words[j+1] = words[j+1], words[j]
                            same = False
                            break
                        elif words[j][k] < words[j+1][k]:
                            output += P.one_iteration_1_1_1_1_1_k_small.format(len(words[j]), len(words[j+1]), min(len(words[j]), len(words[j+1])), k, words[j][k], words[j+1][k])
                            same = False
                            break
                        else:
                            output += P.one_iteration_1_1_1_1_1_k_equal.format(len(words[j]), len(words[j+1]), min(len(words[j]), len(words[j+1])), k, words[j][k], words[j+1][k])
                    # Check If Prefix
                    if same and len(words[j]) > len(words[j+1]):
                        words_ = deepcopy(words)
                        words_[j], words_[j+1] = words_[j+1], words_[j]
                        output += P.check_if_prefix_1_1_1_1_2_enter.format(same, len(words[j]), len(words[j+1]), len(words[j]) > len(words[j+1]), words, words[j], words[j+1], words_)
                        # Swap the words
                        words[j], words[j+1] = words[j+1], words[j]
                    else:
                        output += P.check_if_prefix_1_1_1_1_2_not_enter.format(same, len(words[j]), len(words[j+1]), len(words[j]) > len(words[j+1]), words)
            output += P.return_result.format(words, ' '.join(words))
            return {"input": input,
                    "output": output}

    def final_data(self,data:dict) -> dict:
        if self.name == "coin_flip":
            instruction = "Follow the markov rule to solve the question:\n\nQ: " + data["question"]
            outputs = deepcopy("[initialize]: the coin is heads up.\n")
            state = True
            for idx,flip in enumerate(data["flips"]):
                state = state^flip
                flip_act = "filp" if flip else "does not flip"
                act = "turns" if flip else "remains"
                state_act = "heads up" if state else "tails up"
                outputs += f"[step]: {data['names'][idx]} {flip_act} the coin. So the coin {act} {state_act}.\n"
            return {"input": instruction,
                    "output": outputs}


    

if __name__ == "__main__":
    import json
    from tqdm import tqdm
    from pathlib import Path

    # task = "coin_flip"
    lengths = np.arange(1, 16)
    sample_num_each_length = 2000

    def generate(task):
        Generator = Dataset_Generator(task)
        for length in lengths:
            data_path = Path(__file__).parent/f"{task}/l={length}.json"
            data_path.parent.mkdir(exist_ok=True,parents=True)
            samples = []
            if task == "hyperbaton" and (length == 1 or length > 8):
                continue
            for idx in tqdm(range(sample_num_each_length),desc=f"creating dataset for {task}, l = {length}" ):
                data = Generator.gen_data_from_len(length)
                sample = Generator.final_data(data)
                #print(sample["output"])
                sample["length"] = int(length)
                sample["idx"] = idx
                samples.append(sample)
            utils.dump_json_file(samples,f"{task}/l={length}.json")
            
            #with open(f"{task}/l={length}.json", "w") as f:
                #json.dump(samples, f)

    def gathered(tasks,split_rate=0.02,):
        
        dataset_path = Path(__file__).parent/"output"/f"dataset.jsonl"
        dataset_path.parent.mkdir(parents=True,exist_ok=True)
        data = []
        new_data = []
        for task in tasks:
            data_path = Path(__file__).parent/f"{task}"
            for input_file_path in data_path.glob("*.json"):
                data.extend(utils.load_json_file(input_file_path))
        for d in data:
            conversations = [
                create_user_message(input_type="text",user_prompt=d["input"]),
                create_assistant_message(assistant_prompt=d["output"]),]
            new_d = dict(
                conversations=conversations,
                id = str(d["length"])+"_"+str(d["idx"]),
                length = d["length"],
            )
            new_data.append(new_d)
        
        test_samples = int(min(len(new_data)*split_rate, 1000))
        test_data =new_data[:test_samples]
        train_data = new_data[test_samples:]
        with open(str(dataset_path).replace('.jsonl', '-train.json'), 'w') as f:
            json.dump(train_data, f)
        with open(str(dataset_path).replace('.jsonl', '-valid.json'), 'w') as f:
            json.dump(test_data, f)
        

# for task in ["coin_flip", "last_letter_concat", "reverse_list", "dyck_languages", "navigate", "word_sorting", "object_counting", "hyperbaton",]:
    tasks =  ["coin_flip",]
    for task in tasks:
        generate(task)
    gathered(tasks)

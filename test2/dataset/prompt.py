class Prompt:
    def __init__(self, dataset_name) -> None:
        if dataset_name == "coin_flip":
            self.rule = '''
def coin_flip(flips):
    # Initialize Coin State
    heads_up = True
    # Main Loop
    for flip in flips:
        if flip:
            heads_up = not heads_up
        else:
            pass
    return heads_up'''
            self.initialize = '''
flips = {}
1. Initialize Coin State
heads_up = True
2. Main Loop'''
            self.one_iteration_2_1_flip = '''
2.1 one iteration
```
for flip in flips:
```
flip = True
```
if flip:
```
flip = True
enter
```
heads_up = not heads_up
```
heads_up = {}
heads_up = not heads_up = {}'''
            self.one_iteration_2_1_no_flip = '''
2.1 one iteration
```
for flip in flips:
```
flip = False
```
if flip:
```
flip = False
do not enter
```
else:
    pass
```
heads_up = {}'''
            self.return_result = '''
3. Return Result
```
resturn heads_up
```
heads_up = {}

So the answer is {}.'''

        elif dataset_name == "last_letter_concat":
            self.rule = '''
def last_letter_concat(names):
    # Initialize Result
    result = ""
    # Main Loop
    for name in names:
        result += name[-1]
    return result'''
            self.initialize = '''
names = {}
1. Initialze Result
result = ""
2. Main Loop'''
            self.one_iteration_2_1 = '''
2.1 one iteration
```
for name in names:
```
name = "{}"
```
result += name[-1]
```
result = "{}"
name[-1] = "{}"
result += "{}"
result = "{}"'''

            self.return_result = '''
3. Return Result
```
resturn result
```
result = "{0}"

So the answer is "{0}".'''
       
        elif dataset_name == "reverse_list":
            self.rule = '''
def reverse_list(l):
    # Initialize Result
    result = []
    # Main Loop
    while l:
        result.append(l.pop())
    return result'''
            self.initialize = '''
l = {}
1. Initialze result
result = []
2. Main Loop'''
            self.one_iteration_2_1 = '''
2.1 one iteration
```
while l:
```
l = {0}
```
result.append(l.pop())
```
result = {1}
l.pop() = {2}
result = result.append({2}) = {3}'''

            self.return_result = '''
3. Return Result
```
resturn result
```
result = {}

So the answer is "{}".'''

        elif dataset_name == "object_counting":
            self.rule = '''
PROCEDURE object_counting(objects, target);
    # Initialize Count
    count <- 0;
    # Main Loop
    FOR object <- objects[0] to objects[-1] DO
        IF <<object belong to target>> THEN
            count <- count + object.num;
        ENDIF
    ENDFOR
    # Output Result
    OUTPUT count'''
            self.initialize = '''
objects = {}
target = {}
1. Initialize Count
count = 0
2. Main Loop'''
            self.one_iteration_2_1_class = '''
2.1 one iteration
```
FOR object <- objects[0] to objects[-1] DO
```
object = {0}
```
IF <<object belong to {1}>> THEN
```
{0} belong to {1}
enter
```
count <- count + object.num;
```
count = {2}
object.num = {3}
count = count + object.num = {2} + {3} = {4}'''
            self.one_iteration_2_1_not_class = '''
2.1 one iteration
```
FOR object <- objects[0] to objects[-1] DO
```
object = {0}
```
IF <<object belong to {1}>> THEN
```
{0} do not belong to {1}
do not enter
count = {2}'''

            self.return_result = '''
3. Return Result
```
OUTPUT count
```
count = {0}

So the answer is {0}.'''

        elif dataset_name == "dyck_languages":
            self.rule = '''
def complete(s):
    ketbra_map = {")": "(", "}": "{", "]": "[", ">": "<"}
    # Initialize Stack
    stack = []
    # Stack Loop
    for braket in s:
        if braket in ketbra_map:
            top = stack.pop() if stack else None
            if ketbra_map[braket] != top:
                return "invalid"
        else:
            stack.append(braket)
    # Initialize Complete
    braket_map = {"(": ")", "{": "}", "[": "]", "<": ">"}
    complete = ""
    # Completion Loop
    while stack:
        complete += braket_map[stack.pop()]
    # Return Result
    return complete'''

            self.initialize_1 = '''
s = "{}"'''
            self.initialize_2 = '''
ketbra_map = {")": "(", "}": "{", "]": "[", ">": "<"}
1. Initialize Stack
stack = []
2. Stack Loop'''

            self.one_iteration_2_1_map = '''
2.1 one iteration
```
for braket in s:
```
braket = '{0}'
```
if braket in ketbra_map:
```
'{0}' in [")", "}}", "]", ">"]
enter
```
top = stack.pop() if stack else None
```
stack = {1}
enter
top = stack.pop() = '{2}'
stack = {3}
```
if ketbra_map[braket] != top:
```
ketbra_map[braket] = ketbra_map['{0}'] = '{4}'
top = '{2}'
ketbra_map[braket] == top
do not enter'''
            self.one_iteration_2_1_not_map = '''
2.1 one iteration
```
for braket in s:
```
braket = '{0}'
```
if braket in ketbra_map:
```
'{0}' not in [")", "}}", "]", ">"]
do not enter
```
else:
    stack.append(braket)
```
stack = {1}
braket = '{0}'
stack.append('{0}')
stack = {2}'''

            self.initialize_complete = '''
3. Initialize Complete
braket_map = {"(": ")", "{": "}", "[": "]", "<": ">"}
complete = ""
4. Completion Loop'''

            self.one_iteration_4_1_stack = '''
4.1 one iteration
```
while stack:
```
stack = {0}
enter
```
complete += braket_map[stack.pop()]
```
stack.pop() = '{1}'
braket_map['{1}'] = '{2}'
complete = complete + '{2}' = '{3}' + '{2}' = '{4}'
stack = {5}'''

            self.one_iteration_4_1_not_stack = '''
4.1 one iteration
```
while stack:
```
stack = []
do not enter'''

            self.return_result = '''
5. Return Result
```
return complete
```
complete = '{0}'

So the answer is '{0}'.'''

        elif dataset_name == "hyperbaton":
            self.rule = '''
Correct order of the type of adjectives is:
opinion, size, physical quality, shape, age, colour, origin, material, type, purpose.

PROCEDURE adjective_judge(adjs);
    # Initialize
    current_type <- None;
    correct <- True;
    # Main Loop
    FOR adj <- adjs[0] to adjs[-1] DO
        IF <<adj.type later than current type>> THEN
            current_type <- adj.type;
        ELSE
            correct <- False
            break
        ENDIF
    ENDFOR
    # Output Result
    OUTPUT correct'''
            self.initialize = '''
1. Initialize Type
current_type = None
correct = True
2. Main Loop'''
            self.one_iteration_2_1_later = '''
2.1 one iteration
```
FOR adj <- adjs[0] to adjs[-1] DO
```
adj = {0}
```
IF <<adj.type later than current type>> THEN
```
current_type is {1}
{0} is used to describe {2}
{2} should appear later than {1}
enter
```
current_type <- adj.type;
```
current_type = adj.type = {2}'''
            self.one_iteration_2_1_earlier = '''
2.1 one iteration
```
FOR adj <- adjs[0] to adjs[-1] DO
```
adj = {0}
```
IF <<adj.type later than current type>> THEN
```
current_type is {1}
{0} is used to describe {2}
{2} should appear ealier than {1}
do not enter
```
ELSE
    correct <- False
    break
```
correct = False
end the loop
'''

            self.return_result = '''
3. Return Result
```
OUTPUT correct
```
correct = {}

So the answer is {}.'''

        elif dataset_name == "navigate":
            self.rule = '''
def navigate(moves):
    # Initialize Location
    loc = [0, 0]
    # Main Loop
    for move in moves:
        if move[0] == "left":
            loc[0] -= move[1]
        elif move[0] == "right":
            loc[0] += move[1]
        elif move[0] == "forward":
            loc[1] += move[1]
        elif move[0] == "backward":
            loc[1] -= move[1]
    return loc == [0, 0]'''
            self.initialize = '''
moves = {}
1. Initialze result
loc = [0, 0]
2. Main Loop'''
            self.one_iteration_2_1_left = '''
2.1 one iteration
```
for move in moves:
```
move = {0}
```
if move[0] == "left":
    loc[0] -= move[1]
```
loc = {1}
move[1] = {2}
loc[0] = loc[0] - move[1] = {3} - {2} = {4}
loc = {5}'''

            self.one_iteration_2_1_right = '''
2.1 one iteration
```
for move in moves:
```
move = {0}
```
if move[0] == "right":
    loc[0] += move[1]
```
loc = {1}
move[1] = {2}
loc[0] = loc[0] + move[1] = {3} + {2} = {4}
loc = {5}'''

            self.one_iteration_2_1_forward = '''
2.1 one iteration
```
for move in moves:
```
move = {0}
```
if move[0] == "forward":
    loc[1] += move[1]
```
loc = {1}
move[1] = {2}
loc[1] = loc[1] + move[1] = {3} + {2} = {4}
loc = {5}'''

            self.one_iteration_2_1_backward = '''
2.1 one iteration
```
for move in moves:
```
move = {0}
```
if move[0] == "backward":
    loc[1] -= move[1]
```
loc = {1}
move[1] = {2}
loc[1] = loc[1] - move[1] = {3} - {2} = {4}
loc = {5}'''

            self.return_result = '''
3. Return Result
```
return loc == [0, 0]
```
loc = {}
loc == [0, 0] = {}

So the answer is {}.'''

        elif dataset_name == "word_sorting":
            self.rule = '''
def word_sorting(words):
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
                    break
                elif words[j][k] < words[j+1][k]:
                    same = False
                    break
            # Check If Prefix
            if same and len(words[j]) > len(words[j+1]):
                # Swap the words
                words[j], words[j+1] = words[j+1], words[j]
    return words'''
            self.initialize = '''
words = {}
len(words) = {}
1. Main Loop'''
            self.one_iteration_1_1_i = '''
1.1 one iteration (i)
```
for i in range(n):
```
i = {}'''
            self.one_iteration_1_1_1_j = '''
1.1.1 one iteration (j)
```
for j in range(0, n-i-1):
```
j = {}
words[j] = "{}"
words[j+1] = "{}"
1.1.1.1 Compare Words Letter By Letter
same = True'''
            self.one_iteration_1_1_1_1_1_k_big = '''
1.1.1.1.1 one iteration (k)
```
for k in range(min(len(words[j]), len(words[j+1]))):
```
len(words[j]) = {0}
len(words[j+1]) = {1}
min(len(words[j]), len(words[j+1])) = {2}
k = {3}
```
if words[j][k] > words[j+1][k]:
```
words[j][k] = "{4}"
words[j+1][k] = "{5}"
"{4}" > "{5}"
enter
```
# Swap the words
words[j], words[j+1] = words[j+1], words[j]
same = False
break
```
words = {6}
swap words[j] = "{7}" and words[j+1] = "{8}"
words = {9}
same = False
break'''

            self.one_iteration_1_1_1_1_1_k_small = '''
1.1.1.1.1 one iteration (k)
```
for k in range(min(len(words[j]), len(words[j+1]))):
```
len(words[j]) = {0}
len(words[j+1]) = {1}
min(len(words[j]), len(words[j+1])) = {2}
k = {3}
```
if words[j][k] > words[j+1][k]:
```
words[j][k] = "{4}"
words[j+1][k] = "{5}"
"{4}" < "{5}"
do not enter
```
elif words[j][k] < words[j+1][k]:
```
words[j][k] = "{4}"
words[j+1][k] = "{5}"
"{4}" < "{5}"
enter
```
same = False
break
```
same = False'''

            self.one_iteration_1_1_1_1_1_k_equal = '''
1.1.1.1.1 one iteration (k)
```
for k in range(min(len(words[j]), len(words[j+1]))):
```
len(words[j]) = {0}
len(words[j+1]) = {1}
min(len(words[j]), len(words[j+1])) = {2}
k = {3}

```
if words[j][k] > words[j+1][k]:
```
words[j][k] = "{4}"
words[j+1][k] = "{5}"
"{4}" = "{5}"
do not enter
```
elif words[j][k] < words[j+1][k]:
```
words[j][k] = "{4}"
words[j+1][k] = "{5}"
"{4}" = "{5}"
do not enter'''

            self.check_if_prefix_1_1_1_1_2_enter = '''
1.1.1.1.2 Check If Prefix
```
if same and len(words[j]) > len(words[j+1])::
```
same = {0}
len(words[j]) = {1}
len(words[j+1]) = {2}
len(words[j]) > len(words[j+1]) = {3}
same and len(words[j]) > len(words[j+1]) = {0} and {3} = True
enter
```
# Swap the words
words[j], words[j+1] = words[j+1], words[j]
```
words = {4}
swap words[j] = "{5}" and words[j+1] = "{6}"
words = {7}'''
            self.check_if_prefix_1_1_1_1_2_not_enter = '''
1.1.1.1.2 Check If Prefix
```
if same and len(words[j]) > len(words[j+1])::
```
same = {0}
len(words[j]) = {1}
len(words[j+1]) = {2}
len(words[j]) > len(words[j+1]) = {3}
same and len(words[j]) > len(words[j+1]) = {0} and {3} = False
do not enter
words = {4}'''
            self.return_result = '''
3. Return Result
```
return words
```
words = {}

So the answer is "{}".'''

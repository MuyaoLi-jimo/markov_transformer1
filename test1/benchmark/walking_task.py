import numpy as np
from utils import utils
from pathlib import Path
import copy
import re
direction = [["west","east"],["south","north"]]

def my_move(coo:list):
    idx = np.random.choice([0,1])
    idy = int(np.random.choice([0,1]))
    z = int(np.random.choice(np.arange(10)))
    coo[idx] += (idy*2-1)*z
    instruction = f"Walk {z} meters to the {direction[idx][idy]}."
    return coo,instruction

def produce():
    benchmark_path = Path(__file__).parent / "markov_bench.jsonl"
    jp = utils.JsonlProcessor(benchmark_path)
    for i in range(20):
        steps = i*2+1
        coo = [int(np.random.randint(-10,10)),int(np.random.randint(-10,10))]
        start_state = copy.copy(coo)
        instructions = []
        answers = []
        
        for _ in range(steps):
            coo,instruction = my_move(coo)
            instructions.append(instruction)
            answers.append(copy.copy(coo))
        data = dict(
            id=i+1,
            start=start_state,
            instruction=instructions,
            answer = answers
        )
        jp.dump_line(data)
    jp.close()

def check_task(inputs:str,answer:list):
    #print(inputs,answer)
    pattern = r'\[(-?\d+),\s*(-?\d+)\]'
    matches = re.findall(pattern, inputs)
    if matches and int(matches[-1][0])==answer[0] and int(matches[-1][1])==answer[1]:
        return True
    return False
    

if __name__ == "__main__":
    #print(check_task("[2,2]Bob's current location is at coordinates [15, -6].", [15, -6]))
    produce()
    
    
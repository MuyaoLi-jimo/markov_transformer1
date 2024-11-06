from pathlib import Path
from utils import utils

task = "coin_flip"
length = 3
data_path = Path(__file__).parent/f"{task}/l={length}.json"
dataset = utils.load_json_file(data_path)
print(dataset[0]["input"])
print("____________")
print(dataset[0]["output"])

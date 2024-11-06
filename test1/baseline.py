import json
import os
import requests
from utils import utils
from benchmark import walking_task
from pathlib import Path
import message_generate
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI


OPENAI_SYSTEM = "you are a helpful assistant."
OPENAI_INSTRUCTION = ["Bob starts at the origin and moves "," meters "," and "," meters ",", reaching coordinates ",
                      "In these coordinates, the first value represents the east-west direction, where positive is east and negative is west. The second value represents the north-south direction, where positive is north and negative is south. \nHe then ",
                      "Where is he now? \nPlease provide his current coordinates in the format: coordinates [x, y]."]

def generate_qa_openai(data,model_id = "gpt-4",temperature=0.7,max_tokens=4096):
    
    client = OpenAI()
    model = data.get("model_id",model_id)
    temperature = data.get("temperature",temperature)
    max_tokens = data.get("max_tokens",max_tokens)
    response = client.chat.completions.create(
        model=model,
        messages=data["messages"],
        temperature=temperature,
        max_tokens= max_tokens,
    )
    content = response.choices[0].message.content
    token = {
        "input" : response.usage.prompt_tokens,
        "output" : response.usage.completion_tokens,
    }
    return content,token

def generate_qa_xinyun_proxy(data,model_id = "gpt-4o",temperature=0.7,max_tokens=4096):
    url = "https://www.dwyu.top/v1/chat/completions"
    headers = {
        'Authorization': os.environ.get("TEMP_API"),
        'Accept': 'application/json',
        'User-Agent': 'Apifox/1.0.0 (https://w.com)',
        'Content-Type': 'application/json'
    }
    model = data.get("model_id",model_id)
    #print(model)
    temperature = data.get("temperature",temperature)
    max_tokens = data.get("max_tokens",max_tokens)
    payload = json.dumps({
        "model": model,
        "stream": False,
        "messages": data["messages"],
        "temperature":temperature,
        "max_tokens": max_tokens,
    })
    response = requests.request("POST", url, headers=headers, data=payload)
    response = response.json() 

    try:
        content = response['choices'][0]['message']['content']
    except Exception as e:
        print(response)
        raise Exception
    token = {
        "input": response['usage']['prompt_tokens'],
        "output": response['usage']['completion_tokens'],
    }
    return content,token

def generate_qa_vllm(data:dict,model_id = "gpt-4o",temperature=0.7,max_tokens=2048,
                     openai_api_key = "EMPTY",openai_api_base = "http://localhost:9001/v1",host="localhost"):
    model = data.get("model_id",model_id)
    temperature = data.get("temperature",temperature)
    max_tokens = data.get("max_tokens",max_tokens)
    openai_api_key = data.get("openai_api_key",openai_api_key)
    openai_api_base = data.get("openai_api_base",openai_api_base)
    # if give a port,then just use it
    port = data.get("port",None)
    if port:
        openai_api_base = f"http://{data.get('host',host)}:{port}/v1"
        
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    models = client.models.list()
    model = models.data[0].id
    response = None
    try:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        chat_completion = client.chat.completions.create(
            model=model,
            messages= data["messages"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = chat_completion.choices[0].message.content
    except Exception as e:
        print(e)
    return response

def generate_qa_llama(data,tokenizer,model,temperature=0.7,max_tokens=4096,device="cuda:1"):
    text = tokenizer.apply_chat_template(data["messages"], tokenize=False, add_generation_prompt=False)
    input = tokenizer(text,return_tensors="pt", padding=True, max_length=max_tokens,truncation=True).to(device)
    generate_ids = model.generate(input.input_ids, max_length=max_tokens,temperature=temperature)
    decoded_sequences = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return decoded_sequences

def instruction_make(inputs:dict,model_type)->list:
    direction = [["west","east"],["south","north"]]
    def transform_number(n):
        if n >= 0:
            return 1
        else:
            return 0
    messages = []
    messages.append(message_generate.create_system_message(OPENAI_SYSTEM,model_type=model_type))
    start = inputs.get("start")
    instructions = inputs.get("instruction")
    prompt = OPENAI_INSTRUCTION[0] + str(abs(start[0])) + OPENAI_INSTRUCTION[1] + direction[0][transform_number(start[0])] 
    prompt += OPENAI_INSTRUCTION[2] + str(abs(start[1])) + OPENAI_INSTRUCTION[3] + direction[1][transform_number(start[1])] 
    prompt += OPENAI_INSTRUCTION[4] + str(start) + OPENAI_INSTRUCTION[5] 
    for instruction in instructions:
        prompt += instruction + " \n"
    prompt += OPENAI_INSTRUCTION[6]
    messages.append(message_generate.create_user_message(input_type="text",user_prompt=prompt,model_type=model_type))
    return messages 

def qa_wrapper(inputs,model_type="gpt-4",):
    messages = instruction_make(inputs,model_type)
    data = dict(
        messages=messages,
        temperature=0.7,
        model_id=model_type,
    )
    #content,_ = generate_qa_openai(data)
    data["port"]=9001
    content = generate_qa_vllm(data)
    return content,walking_task.check_task(content,inputs["answer"][-1])


def multistep_markov(inputs,model_type="gpt-4",):
    """stoped at every '.' """
    messages = instruction_make(inputs,model_type)
    import copy
    raw_messages = copy.copy(messages)
    for i in range(2*len(inputs["instruction"])):
        data = dict(
            messages=messages,
            temperature=0.7,
            port=9001,
        )
        content = generate_qa_vllm(data)
        contents = []
        for cs in content.split("."):
            for c in cs.split("\n"):
                if c==" "or c=="" or len(c)<3:
                    continue
                contents.append(c) 
        print(content,len(contents),contents)
        if walking_task.check_task(content,inputs["answer"][-1]):
            return str(raw_messages),True
        if len(contents)>1:
            step_output = contents[0]+contents[1]
        else:
            return str(raw_messages),walking_task.check_task(str(raw_messages),inputs["answer"][-1])
        messages.append(message_generate.create_assistant_message(step_output))
        messages.append(message_generate.create_user_message(input_type="text",user_prompt="then?"))
        raw_messages.append(message_generate.create_assistant_message(step_output))
        raw_messages.append(message_generate.create_user_message(input_type="text",user_prompt="then?"))
        if len(messages)>=8:
            messages[5]=messages[7]
            messages = messages[:-2]
    
    return str(raw_messages),walking_task.check_task(str(raw_messages),inputs["answer"][-1])

def test():
    benchmark_path = Path(__file__).parent /"benchmark"/ "markov_bench.jsonl"
    record_path = Path(__file__).parent / "record" /"record.jsonl"
    jp = utils.JsonlProcessor(benchmark_path)
   
    record_jp = utils.JsonlProcessor(record_path)
    record_jp.dump_restart()
    
    #device="cuda:1"
    #tokenizer = AutoTokenizer.from_pretrained("/nfs-shared/models/llama-3/llama-3-8b-instruct")
    #model = AutoModelForCausalLM.from_pretrained("/nfs-shared/models/llama-3/llama-3-8b-instruct")
    #model.to(device)
    
    while True:
        line = jp.load_line()
        if not line:
            break
        print(line["id"])
        content,reward=multistep_markov(line,model_type="llama-2")
        outputs = dict(
            id=line["id"],
            reward=reward,
            content=content,
        )
        record_jp.dump_line(outputs)
        
if __name__ == "__main__":
    test()
from pathlib import Path
from typing import Literal
from utils import utils

ANTHROPIC_MODEL_LIST = {
    "claude-1",
    "claude-2",
    "claude-2.0",
    "claude-2.1",
    "claude-3-haiku-20240307",
    "claude-3-haiku-20240307-vertex",
    "claude-3-sonnet-20240229",
    "claude-3-sonnet-20240229-vertex",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-instant-1",
    "claude-instant-1.2",
}

OPENAI_MODEL_LIST = {
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-browsing",
    "gpt-4-turbo-2024-04-09",
    "gpt2-chatbot",
    "im-also-a-good-gpt2-chatbot",
    "im-a-good-gpt2-chatbot",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o",
    "gpt-4o-mini",
    "chatgpt-4o-latest-20240903",
    "chatgpt-4o-latest",
    "o1-preview",
    "o1-mini",
}

def create_system_message(system_prompt,model_type="jarvis-train"):
    message = None
    if model_type in OPENAI_MODEL_LIST or model_type in ANTHROPIC_MODEL_LIST:
        message = {
            "role": "system",
            "content": system_prompt,
        }
    else:
        raise ValueError(f"don't define the model_id: {model_type}")
    return message

def create_assistant_message(assistant_prompt,model_type="jarvis-train"):
    message = None
    if model_type =="jarvis-train":
        message = {
            "role": "assistant",
            "content": [{
                    "type": "text",
                    "text": f"{assistant_prompt}"
                }],
        }
        return message
    message = {
        "role": "assistant",
        "content": assistant_prompt,
    }
    return message

def create_user_message(input_type:Literal["text","image"],user_prompt,source_data=[],model_type="jarvis-train"):
    message = None
    if model_type == "jarvis-train":
        if input_type=="text":
            message = {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"{user_prompt}"
                }],
            }
        elif input_type=="image":
            message = {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"{user_prompt}"
                },
                {
                    "type": "image",
                    "text": "<image>",
                },
                ],
            }
        else:
            raise AssertionError(f"type error:{input_type}")
        return message
    
    if input_type=="text":
        message = {
            "role": "user",
            "content": user_prompt,
        }
    elif input_type=="image":
        source_data = Path(source_data[0])
        if model_type in { "MiniCPM-V-2_6" ,"llava-1.5-13b-hf", "llava-1.5-7b-hf", "fuyu-8b","llava-v1.6-mistral-7b-hf"} or "molmo" in model_type:
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{user_prompt}\n"
                    },
                    {
                        "type": "image_url",
                        "image_url": { "url": f"data:image/{str(source_data.suffix)[1:]};base64,{utils.encode_image_to_base64(source_data)}"},
                    },
                ]
            }
        elif model_type in ANTHROPIC_MODEL_LIST:
            image_suffix = "jpeg" if str(source_data.suffix)[1:]=="jpg" else str(source_data.suffix)[1:]
            message = {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"<image>{user_prompt}\n"
                },
                {
                    "type": "image_url",
                    "image_url": { "url": f"data:image/{image_suffix};base64,{utils.encode_image_to_base64(source_data)}"},
                },
                ],
            }
        else:
            message = {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"<image>{user_prompt}\n"
                },
                {
                    "type": "image_url",
                    "image_url": { "url": f"data:image/{str(source_data.suffix)[1:]};base64,{utils.encode_image_to_base64(source_data)}"},
                },
                ],
            }
    else:
        raise AssertionError(f"type error:{input_type}")

    return message

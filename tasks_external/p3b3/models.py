from dataclasses import dataclass
from dotenv import load_dotenv
from google.genai.types import GenerateContentConfig
from openai import OpenAI
from google.genai.errors import ServerError
import google.generativeai as genai
from itertools import batched
# from dotenv import load_dotenv
from google.genai.types import HttpOptions
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, cast, TypeAlias, TypedDict, Literal, Union
from tqdm import tqdm

import os, time, sys

# Load env
load_dotenv()

MAX_RETRIES = 5
# MAX_RETRIES = 1

@dataclass
class GenerationResult:
    outputs : list[str]
    raw_outputs : list[str]

class EntityTurn(TypedDict):
    role: Literal['user', 'assistant', 'system']
    content: str

TurnedPrompt   = list[EntityTurn]
BatchedPrompts = list[str] | list[TurnedPrompt]

class ResponseIsNone(Exception): 
    def __init__(self): pass
    def __str__(self): return 'model response is none'

def parallel_generation(
        func : Callable[[Any], str],
        prompts : list[Any], 
        max_connections: int
    ) -> list[str]:

    def wrapper(prompt : str) -> str:
        res = func(prompt)
        if not res:
            raise ResponseIsNone()
        return res

    def retry(prompt : str) -> str:
        for i in range(MAX_RETRIES):
            try: return wrapper(prompt)
            # TODO stop excepting Exception, be more specific
            except Exception as ex:
                times = i + 1
                print("Error doing API call :", ex, file=sys.stderr)
                print(f"RETRY {times:02}/{MAX_RETRIES}", file=sys.stderr)
                time.sleep(1.5 * times)

        # TODO: property handle gemini 'request quota'
        return wrapper(prompt)

    with ThreadPoolExecutor(max_workers=max_connections) as executor:
        return [result for result in tqdm(executor.map(retry, prompts), total=len(prompts))]

class Model:
    def get_name(self) -> str: ...
    def generate_in_batch(self, prompts: BatchedPrompts, max_connections : int) -> list[str]: ...
    def generate(self, prompts: BatchedPrompts) -> list[str]: 
        return self.generate_in_batch(prompts, max_connections=10)
    def chat(self, messages) -> str:
        return self.generate([messages])[0]
    def generate_with_debug(self, prompts: BatchedPrompts, max_connections : int) -> GenerationResult:
        outputs = self.generate_in_batch(prompts, max_connections) # pyright: ignore
        return GenerationResult(
            outputs, raw_outputs=[
               (
                   f'user:\n{ctx}' if isinstance(ctx, str) else \
                       '\n\n'.join(f'{p['role']}:\n{p['content']}' for p in ctx)
               ) + '\n\nassistant:\n' + output  for ctx, output in zip(prompts, outputs)
            ]
        )

class ChatGPT(Model):
    def __init__(self, model_name : str):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )

    def get_name(self) -> str:
        return self.model_name

   
    def generate_in_batch(self, prompts: BatchedPrompts, max_connections : int) -> list[str]:
        assert len(prompts), 'prompts list should not be empty'

        batched_messages = prompts if isinstance(prompts[0], list) else \
            [[{'role': 'user', 'content': prompt}] for prompt in prompts]

        return parallel_generation(lambda messages: 
            self.client.chat.completions.create(
                model=self.model_name, messages=messages, # pyright: ignore
                temperature=0
            ).choices[0].message.content, # pyright: ignore
            batched_messages, max_connections
        ) 

class Gemini(Model):
    def __init__(self, model_name : str):
        self.model_name = model_name
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.client     = genai.GenerativeModel(
            model_name=model_name,
            # http_options=HttpOptions(timeout=300_000), # 4 minutes per request
        )

    def get_name(self) -> str:
        return self.model_name

    def convert_role(self, role : str) -> str:
        if role == 'assistant': return 'model'
        if role == 'user': return 'user'
        raise Exception(f"Gemini doesn't support the role '{role}'")
    
    def chat(self, messages: Union[str, TurnedPrompt]) -> str:
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        response = self.client.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.5)
        )
        return response.text
    
    def generate_in_batch(self, prompts: BatchedPrompts, max_connections : int) -> list[str]:
        if isinstance(prompts[0], str):
            prompts = cast(list[str], prompts)
            return parallel_generation(lambda prompt: 
                self.client.models.generate_content(
                    model=self.model_name, contents=prompt,
                    config=GenerateContentConfig(temperature=0, max_output_tokens=1024) # in Iago's words: to make it more stable (the temperature=0)
                ).text, # pyright: ignore
                prompts, max_connections
            ) 
        else:
            prompts = cast(list[TurnedPrompt], prompts)
            processed_prompts = [
              [ 
               {'role': self.convert_role(entry['role']), 'parts': [{'text': entry['content']}]} 
                for entry in conv
              ] for conv in prompts
            ]

            def process_request(messages):
                history = messages[:-1]
                prompt  = messages[-1]

                assert prompt['role'] == 'user', 'last turn should be `user`'
                chat = self.client.chats.create(model=self.model_name, history=history)

                print(prompt)
                assert len(prompt['parts']) == 1

                return chat.send_message(
                    prompt['parts'][0]['text'],
                    config= GenerateContentConfig(
                        temperature=0, max_output_tokens=1024, top_p=1
                    )
                ).text

            return parallel_generation(process_request, processed_prompts, max_connections) # pyright: ignore

class HuggingFaceModel(Model):
    def __init__(self, model_name : str, system_prompt : str | None = None, **gen_args):
        from vllm import LLM
        # self.max_connections = max_connections
        self.model_name = model_name
        self.model = LLM(model=model_name, seed=42)
        self.extra_msgs : TurnedPrompt = [{"role": "system", "content": system_prompt }] if system_prompt else []
        self.get_args = gen_args

    def get_name(self) -> str:
        return self.model_name

    # def generate_with_debug(self, prompts: list[PromptType], max_connections : int = 10) -> GenerationResult:
    def generate_with_debug(self, prompts: BatchedPrompts, max_connections : int) -> GenerationResult:
        assert len(prompts) > 0, "list of prompts should not be empty"

        chats = prompts if isinstance(prompts[0], list) else \
            [[{"role": 'user', 'content': prompt }] for prompt in prompts]

        tokenizer = self.model.get_tokenizer()
        processed_prompts = cast(list[str], [
            tokenizer.apply_chat_template(
                self.extra_msgs + chat, # pyright: ignore
                tokenize=False, add_generation_prompt=True
            ) for chat in chats
        ])

        results = self.model.generate(
            processed_prompts, **self.get_args 
        )

        # results = []
        # for batch in batched(processed_prompts, max_connections):
        #     results.extend(self.model.generate(
        #         batch, **self.get_args 
        #     ))

        # print(results[0].outputs[0])
        # print('-' * 100)
        # print(tokenizer.decode(list(results[0].outputs[0].token_ids)))
        # sys.exit(1)
        return GenerationResult(
            outputs = [v.outputs[0].text for v in results],
            raw_outputs = [p + tokenizer.decode(list(v.outputs[0].token_ids)) for p, v in zip(processed_prompts, results)]
        )

    def generate_in_batch(self, prompts: BatchedPrompts, max_connections : int) -> list[str]:
        return self.generate_with_debug(prompts, max_connections).outputs


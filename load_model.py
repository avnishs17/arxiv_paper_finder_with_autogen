from autogen_ext.models.openai import OpenAIChatCompletionClient


class ModelClient(OpenAIChatCompletionClient):    
    def __init__(self, model_name='gpt-4o'):
        super().__init__(model=model_name)  
        
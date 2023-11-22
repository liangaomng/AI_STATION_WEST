

import openai

import os
openai.api_key = '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'

class ChatBot:
    def __init__(self, proxy, model):
        self.model = model
        self.messages = []
        openai.proxy=proxy

    def add_message(self, role, content):
        self.messages.append({'role': role, 'content': content})

    def get_response(self):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
        )
        self.add_message('assistant', response['choices'][0]['message']['content'])

    def print_conversation(self):
        for message in self.messages:
            print(f"{message['role'].title()}: {message['content']}")

# 使用示例
bot = ChatBot(proxy="http://127.0.0.1:4780", model="gpt-3.5-turbo")
bot.add_message('system', 'you are a expert in system identification and sybmolic regression')
bot.add_message('system', 'you will receive a string about (x,y) and tell m ze which basis function in your experience')
bot.add_message('system', 'you could add as much as basis function and just give me your answer like this form : {}, for example {x0},{x2},{x3} ')
bot.add_message('user', '(0,0.52),(1,2),(100,44)')
bot.get_response()
bot.print_conversation()

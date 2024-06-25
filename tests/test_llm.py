#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: test_llm.py 
@time: 2024-06-25
@contact: danerlt001@gmail.com
@desc: 
"""
import pytest
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def test_deepseek_openai():
    client = OpenAI()

    query = "中秋节吃什么"
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": query},
        ],
        stream=False
    )

    answer = response.choices[0].message.content
    print(answer)


if __name__ == '__main__':
    pytest.main()

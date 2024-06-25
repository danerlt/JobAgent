#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: test_have_json.py 
@time: 2024-06-22
@contact: danerlt001@gmail.com
@desc: 
"""
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts import (
    PromptTemplate,
)
from llama_index.core.prompts.utils import get_template_vars


def test_template_hava_json() -> None:
    """Test partial format."""
    prompt_txt = 'hello {text} {foo} \noutput format:\n```json\n{"name": "llamaindex"}\n```'
    except_prompt = 'hello world bar \noutput format:\n```json\n{"name": "llamaindex"}\n```'
    print(prompt_txt)

    prompt_template = PromptTemplate(prompt_txt)

    prompt_fmt = prompt_template.partial_format(foo="bar")
    prompt = prompt_fmt.format(text="world")
    print(prompt)
    assert isinstance(prompt_fmt, PromptTemplate)

    assert prompt == except_prompt

    assert prompt_fmt.format_messages(text="world") == [
        ChatMessage(content=except_prompt, role=MessageRole.USER)
    ]


def test_template_hava_json_2() -> None:
    """Test template with curly braces."""
    prompt_txt = 'hello {text} {foo} \noutput format:\n```json\n{"name": "llamaindex"}\n```'
    except_prompt = 'hello world bar \noutput format:\n```json\n{"name": "llamaindex"}\n```'
    print(prompt_txt)

    prompt_template = PromptTemplate(prompt_txt)

    # Ensure all placeholders are provided
    template_vars = get_template_vars(prompt_txt)
    assert template_vars == ["text", "foo"]

    prompt_fmt = prompt_template.partial_format(foo="bar")
    assert isinstance(prompt_fmt, PromptTemplate)

    prompt = prompt_fmt.format(text="world")
    print(prompt)
    assert prompt == except_prompt

    assert prompt_fmt.format_messages(text="world") == [
        ChatMessage(content=except_prompt, role=MessageRole.USER)
    ]
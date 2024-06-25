#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: test_aliyun_embeding.py 
@time: 2024-06-25
@contact: danerlt001@gmail.com
@desc: 
"""

import os
from http import HTTPStatus

import pytest
import dashscope

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def test_embed_with_str():
    resp = dashscope.TextEmbedding.call(
        model=dashscope.TextEmbedding.Models.text_embedding_v1,
        api_key=os.getenv('DASHSCOPE_API_KEY'),  # 如果您没有配置环境变量，请将您的APIKEY填写在这里
        input='衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买')
    if resp.status_code == HTTPStatus.OK:
        print(resp)
        assert resp is not None
    else:
        print(resp)
        raise Exception(resp)


def test_langchain_embed_with_str():
    from langchain_community.embeddings import DashScopeEmbeddings
    embd = DashScopeEmbeddings(model="text-embedding-v2")
    resp = embd.embed_query("衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买")
    print(resp)
    assert resp is not None


if __name__ == '__main__':
    pytest.main()

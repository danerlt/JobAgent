#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: langchain_react_agent_demo.py 
@time: 2024-06-25
@contact: danerlt001@gmail.com
@desc: 
"""
# Import relevant functionality
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

# Create the agent
memory = SqliteSaver.from_conn_string(":memory:")
model = ChatOpenAI(model="deepseek-chat", temperature=0)
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="你好，我居住在重庆")]}, config
):
    print(chunk)
    print("----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="重庆有什么好吃的？")]}, config
):
    print(chunk)
    print("----")
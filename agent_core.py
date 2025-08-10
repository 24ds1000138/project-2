# agent_core.py
from langchain_experimental.plan_and_execute import PlanAndExecute, load_chat_planner, load_agent_executor
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import AgentOutputParser, Tool
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
import os
import json
import re
import time
from typing import Union
from chat_proxy import ChatAIProxy

class ReActOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        print(f"Parsing LLM output: {text}")

        if "Final Answer:" in text:
            final_answer = text.split("Final Answer:")[-1].strip()
            return AgentFinish(return_values={"output": final_answer}, log=text)

        action_json_match = re.search(r"Action\s*:\s*```json(.*?)```", text, re.DOTALL)
        if action_json_match:
            json_str = action_json_match.group(1).strip()
            try:
                action_dict = json.loads(json_str)
                action = action_dict.get("action")
                action_input = action_dict.get("action_input")
                if action and action_input is not None:
                    return AgentAction(tool=action, tool_input=action_input, log=text)
                else:
                    raise ValueError("Action JSON is missing 'action' or 'action_input'")
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error in Action block: {e}")
                return AgentFinish(return_values={"output": text.strip()}, log=text)
            except ValueError as e:
                print(f"ValueError in Action block: {e}")
                return AgentFinish(return_values={"output": text.strip()}, log=text)

        print(f"Warning: No valid action or final answer found. Treating as final answer.")
        return AgentFinish(return_values={"output": text.strip()}, log=text)

def handle_query(question: str, attachments: dict = None):
    # Added debug logger at the beginning
    print(f"DEBUG: Question in human-readable format: {question}")

    preferred_field_names = {"questions.txt", "question", "questions", "question.txt"}
    file_paths = {}
    if attachments:
        for field, info in attachments.items():
            if field.strip().lower() in preferred_field_names:
                continue
            filename = info.get("filename")
            if filename and 'bytes' in info:
                path = os.path.join('.', filename)
                with open(path, 'wb') as f:
                    f.write(info['bytes'])
                file_paths[filename] = path

    llm = ChatAIProxy(
        model_name="llama3-70b-8192",
        temperature=0
    )
    python_repl = PythonREPLTool()
    url_match = re.search(r'(https?://[^\s]+)', question)
    url_context = ""
    if url_match:
        extracted_url = url_match.group(1)
        url_context = f"The relevant dataset is located at the following URL: {extracted_url}\n"
        question = question.replace(extracted_url, extracted_url)  # just ensure consistent formatting
    python_tool = Tool(
        name="PythonREPL",
        func=python_repl.run,
        description="Executes Python code to read, analyze, and visualize data from files. For example, use pandas.read_csv('./data.csv') for CSVs or matplotlib for visualizations."
    )

    tools = [python_tool]

    SYSTEM_PROMPT = """
    You are a highly specialized data analysis agent. Your single objective is to execute the user's provided task.

    **ABSOLUTELY CRITICAL DIRECTIVES:**

    You are an expert data analyst and Python programmer.
    You have access to a Python REPL where you can run Python code.
    You can load and use any uploaded attachments provided in this task.
    When scraping or fetching data, ALWAYS use the exact URL provided in the user request.
    Never use `input()` or ask the user for any additional information — all code must be non-interactive.
    Do not invent URLs — if the question includes a URL, use it exactly as given.
    When producing images, return them as base64 PNG data URIs under 100,000 characters.
    Your output MUST strictly follow the response format specified in the user question.
    Do not add extra commentary unless explicitly requested.\
    If a file is provided, it will be available in the current working directory with its original filename.
    """

    attachments_info = ""
    if file_paths:
        file_list = "\n".join([f"- {name}: {path}" for name, path in file_paths.items()])
        attachments_info = f"\nAvailable attachments (saved to disk; these are data sources for analysis):\n{file_list}\nUse Python code to open and analyze these files (e.g., pandas.read_csv('./data.csv'))."

    SYSTEM_PROMPT = SYSTEM_PROMPT.format(tool_names=", ".join([tool.name for tool in tools])) + attachments_info

    PLANNER_SYSTEM = SYSTEM_PROMPT + """
    \n\nLet's first understand the user request and devise a plan to solve it. Output the plan starting with the header 'Plan:' followed by a numbered list of steps. Focus ONLY on the user request and any listed attachments. Do not consider 'question.txt', 'questions.txt', 'UploadFile', 'Headers', or form-data as data sources. **Remember, a user request has been provided, do not ask for one.** At the end of the plan, say '<END_OF_PLAN>'
    """

    planner = load_chat_planner(llm)
    executor = load_agent_executor(
        llm=llm,
        tools=tools,
        verbose=True
    )
    if hasattr(executor, "agent"):
        executor.agent.output_parser = ReActOutputParser()

    agent_executor = PlanAndExecute(
        planner=planner,
        executor=executor,
        verbose=True
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = agent_executor.invoke({"input": question})
            break
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                match = re.search(r"Please try again in ([\d.]+)s", str(e))
                wait_time = float(match.group(1)) if match else 1.0
                print(f"Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    raise Exception("Rate limit exceeded after max retries")
            else:
                print(f"Agent error: {e}")
                raise e

    for path in file_paths.values():
        if os.path.exists(path):
            os.remove(path)

    return result
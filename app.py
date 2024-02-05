import streamlit as st
import pandas as pd
import json
import openai
from openai import OpenAI
import os
import re
import matplotlib.pyplot as plt
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import ast

try:
    from Config import OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
except ModuleNotFoundError:
    pass


def csv_agent_func(df: pd.DataFrame, user_message: str, prompt_: str) -> str:
    """Main csv agent function.
    Refer to documentation here: https://python.langchain.com/docs/integrations/toolkits/pandas"""

    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=os.environ["OPENAI_API_KEY"]),
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        temperature=0.1,
        max_tokens=2800,
    )

    full_input = prompt_.format(user_message=user_message, chat_memory=st.session_state['chat_memory'])

    try:
        response = agent.run(full_input)
        return response
    except Exception as e:
        st.write(f"Something went wrong. Please try again.")
        return None


def explain_plot(response: str, prompt_: str) -> str:
    """CSV agent will sometimes create plots, but will not do a good job of explaning it.
    Separate call to create better explanation for plot if necessary"""

    pre_message = prompt_.format(response=response)
    completion = client.chat.completions.create(
    model="gpt-4",
    temperature=0,
    messages=[
        {"role": "user", "content": pre_message},
    ],
    )

    res = completion.choices[0].message.content

    return res


def summarize_error(query: str, colnames: list, prompt_:str):
    """Call when csv agent throws an error. Intended to help users modify their input
    to get an errorless responses"""
    pre_message = prompt_.format(query=query, colnames=colnames)
    completion = client.chat.completions.create(
    model="gpt-4",
    temperature=0,
    messages=[
        {"role": "user", "content": pre_message},
    ],
    )

    res = completion.choices[0].message.content

    return res


@st.cache_resource
def prep_lang_content(lang: str) -> dict:
    """Select chatbot language"""
    with open(os.path.join('languages', lang, 'contents.json'), 'r') as json_file:
        lang_contents = json.load(json_file)
    print(lang_contents)
    return lang_contents


def summarize_chat_history(new_input: str, new_output: str, lang_contents: dict) -> str:
    """Separate call to create a single string that summarizes the whole chat interaction history so far"""
    full_input = lang_contents["summarize_chat_history"].format(chat_memory=st.session_state['chat_memory'], new_input=new_input, new_output=new_output)
    completion = client.chat.completions.create(
    model="gpt-4",
    temperature=0,
    messages=[
        {"role": "user", "content": full_input},
    ],
    )

    res = completion.choices[0].message.content

    return res


def extract_code_from_response(response: str) -> str:
    """Extracts Python code from a string response."""
    try:
        code_pattern = r"```python(.*?)```"
        found = re.findall(code_pattern, response, re.DOTALL)
        if found:
            executeables = "\n".join([w.strip() for w in found])
            return executeables
    except:
        pass
    return None


def initialize_session_states() -> None:
    if 'chat_memory' not in st.session_state:
        # Will hold summarized memory of past chat
        st.session_state['chat_memory'] = "No previous interaction."
    if 'chat_memory_chain' not in st.session_state:
        # Will hold full past chat history
        st.session_state['chat_memory_chain'] = []


def update_chat_memories(user_input_simple: str, summary: str, lang_contents: dict) -> None:
    st.session_state['chat_memory'] = summarize_chat_history(user_input_simple, summary, lang_contents=lang_contents)
    st.session_state['chat_memory_chain'].append({"role":"user", "output":user_input_simple})
    st.session_state['chat_memory_chain'].append({"role":"assistant", "output":summary})


def main_csv_app() -> None:
    initialize_session_states()

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8, gap="small")

    with col8:
        available_languages = ["English", "Spanish"]
        language = st.selectbox("Language", available_languages, index=available_languages.index('English'))

    lang_contents = prep_lang_content(language)

    st.title(lang_contents['chatbot_name'])
    st.write("---")

    file_path = os.path.join('languages', language, "school_comparison_dataframe.csv")
    df = pd.read_csv(file_path)

    st.write(lang_contents['intro_1'])


    st.markdown("  \n".join([w.strip() for w in lang_contents['intro_2'].splitlines()]))
    st.write("---")
    dropdown = st.selectbox(lang_contents['suggestion'],
    lang_contents['topics'],
    index=None,
    placeholder=lang_contents["select_topic_dropdown_text"],)

    if dropdown is not None:
        with open(os.path.join('languages', language, f"{dropdown}_saved.txt"), 'r', encoding="utf-8") as f:
            content = f.read()
        st.write(content)
    st.write("---")
    simple_response_tab, chat_history_tab = st.tabs([lang_contents["simple_response_tab"], lang_contents["chat_history_tab"]])
    with simple_response_tab:
        with st.form("simple"):
            user_input_simple = st.text_input(lang_contents['user_input_simple'])
            simple_response_submit = st.form_submit_button(lang_contents["simple_response_submit"])

            if simple_response_submit:
                with st.spinner(lang_contents['thinking']):
                    response = csv_agent_func(df, user_input_simple, lang_contents['full_input_template'])
                    code_to_execute = extract_code_from_response(response)

                    if code_to_execute:
                        try:
                            # Make data frame available for execution in the context
                            exec(code_to_execute, globals(), {"df": df, "plt": plt})
                            fig = plt.gcf()
                            if fig.get_axes():
                                st.write(fig)
                            summary = explain_plot(response, prompt_=lang_contents['pre_message'])
                            st.write(summary)
                            update_chat_memories(user_input_simple, summary, lang_contents)

                        except Exception as e:
                            summary = summarize_error(user_input_simple, list(df.columns), prompt_=lang_contents['error_pre'])

                    else:
                        st.write(response)
                        summary = response
                        update_chat_memories(user_input_simple, summary, lang_contents)

    st.divider()

    with chat_history_tab:
        st.write(lang_contents["conversation_history_info"])
        if st.session_state['chat_memory_chain'] != []:
            for mem in st.session_state['chat_memory_chain']:
                with st.chat_message(mem['role']):
                    st.write(mem['output'])


if __name__ == "__main__":
    client = OpenAI()
    st.set_page_config(layout="wide")
    main_csv_app()

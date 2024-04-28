import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers 

## Function To get response from LLAma 2 model

def getLLamaresponse(input_text, platform, target_audience, word_count):

  ### LLama2 model
  llm = CTransformers(model='models\\llama-2-7b-chat.ggmlv3.q8_0.bin',
                       model_type='llama',
                       config={'max_new_tokens': 256,
                              'temperature': 0.01})

  ## Prompt Template
  template = """
    Create a social media post for {platform} targeting {target_audience}  about {input_text}.
    Aim for a word count of around {word_count} words.
  """

  prompt = PromptTemplate(input_variables=["platform", "target_audience", "input_text", "word_count"],
                          template=template)

  ## Generate the response from the LLAma 2 model
  response = llm.invoke(prompt.format(platform=platform, target_audience=target_audience, input_text=input_text, word_count=word_count))
  print(response)
  return response


st.set_page_config(page_title="Generate Social Media Content",
                    page_icon='',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Social Media Content ")

input_text = st.text_input("Enter your message/topic")

## Creating columns for additional fields
col1, col2, col3 = st.columns([3, 3, 3])  

with col1:
  platform = st.selectbox('Platform', ('Twitter', 'Instagram', 'Facebook'), index=0)

with col2:
  target_audience = st.selectbox('Target Audience', ('Tech Enthusiasts', 'General Audience', 'Specific Industry'), index=0)


with col3:
  word_count = st.number_input("Desired Word Count", min_value=1, max_value=280, value=50, step=5)  

submit = st.button("Generate")

## Final response
if submit:
  st.write(getLLamaresponse(input_text, platform, target_audience, word_count))

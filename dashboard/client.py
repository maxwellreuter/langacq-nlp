import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OpenAI_key"])

def name_topic(words: str):
    prompt = f"The following words represent a topic: {words}. Can you suggest a concise topic name for this? Just gave me the topic name."
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    topic_name = completion.choices[0].message.content
    return topic_name
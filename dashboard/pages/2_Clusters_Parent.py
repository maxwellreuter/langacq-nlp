import streamlit as st
import pandas as pd
import time
import os

st.title("Clusters")
study = st.selectbox("Choose Study", ['Champaign', 'HSLLD'])

html_content1 = """
<div style="display: flex; justify-content: space-between; font-size: 24px; font-weight: bold; margin-bottom: 28px;">
    <span style="text-align: left;">Baby Interaction</span>
    <span style="text-align: right;">Parent Reassurance</span>
</div>
"""

html_content2 = """
<div style="display: flex; justify-content: space-between; font-size: 24px; font-weight: bold; margin-top: 0px;">
    <span style="text-align: left;">Wild Animals</span>
    <span style="text-align: right;">Children's Stories</span>
</div>
"""
if study == "Champaign":
    st.markdown(html_content1, unsafe_allow_html=True)

else:
    st.markdown(html_content2, unsafe_allow_html=True)

column_colors = {
    "female": "#FF69B4",                  # Example hex color for "female"
    "male": "#1E90FF",                    # Example hex color for "male"
    "female w/o transcripts": "#FFB6C1", # Example hex color for "female w/o transcripts"
    "male w/o transcripts": "#ADD8E6"    # Example hex color for "male w/o transcripts"
}

def display_image(container, age):
    # Define the image path
    path = f"results/{study}/scatter/{age}.csv"
    
    if os.path.exists(path):
        df = pd.read_csv(path)
        pivot_df = df.pivot_table(index="x", columns="gender", values="y", aggfunc="mean")
        container.scatter_chart(pivot_df, x_label=f'Ages in Months: {age}', color=[column_colors[col] for col in pivot_df.columns])
        return True
    return False

# Placeholder for displaying images
image_container = st.empty()

# Default to show age 20
if study == "Champaign":
    if "current_age" not in st.session_state:
        st.session_state["current_age"] = 20

    if "previous_age" not in st.session_state:
        st.session_state["previous_age"] = 20
else:
    if "current_age" not in st.session_state:
        st.session_state["current_age"] = 44

    if "previous_age" not in st.session_state:
        st.session_state["previous_age"] = 44

min_value = 20 if study == "Champaign" else 44

# Sidebar age selection
age = st.sidebar.number_input("Select an age to display", min_value=min_value, max_value=144, step=1)

# Check if the age has changed
if age != st.session_state["current_age"]:
    st.session_state["previous_age"] = st.session_state["current_age"]
    st.session_state["current_age"] = age
    if not display_image(image_container, age):
        st.warning("The selected age does not have an data.")

# Show button to trigger the current image
if st.sidebar.button("Show"):
    if not display_image(image_container, st.session_state["current_age"]):
        st.warning("The selected age does not have data.")

# GIF loop
frequency = st.sidebar.slider("Frequency (images per second)", min_value=1, max_value=10, step=1, value=2)
if st.sidebar.button("Start Loop"):
    while True:  # Infinite loop to keep cycling through images
        for age in range(20, 144):
            if display_image(image_container, age):
                time.sleep(1 / frequency)  # Delay based on the frequency


html_content3 = """
<div style="display: flex; justify-content: space-between; font-size: 24px; font-weight: bold; margin-top: 28px;">
    <span style="text-align: left;">Daily Activities</span>
    <span style="text-align: right;">Caregiving Needs</span>
</div>
"""

html_content4 = """
<div style="display: flex; justify-content: space-between; font-size: 24px; font-weight: bold; margin-top: 0px;">
    <span style="text-align: left;">Conversation Dialogue</span>
    <span style="text-align: right;">Hungry Caterpillar</span>
</div>
"""
if study == "Champaign":
    st.markdown(html_content3, unsafe_allow_html=True)

else:
    st.markdown(html_content4, unsafe_allow_html=True)


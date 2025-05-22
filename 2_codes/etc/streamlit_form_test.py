import streamlit as st
import json

# 一个form示例
with st.form("my_form"):
    st.write("Inside the form")
    slider_val = st.slider("Form slider")
    checkbox_val = st.checkbox("Form checkbox")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        form_data = {"slider_value": slider_val, "checkbox_value": checkbox_val}
        st.json(form_data)

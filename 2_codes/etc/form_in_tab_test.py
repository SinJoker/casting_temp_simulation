import streamlit as st


def test_form_in_tab():
    st.title("表单在Tab中的测试")

    # 测试1: 普通表单
    with st.expander("测试1: 普通表单"):
        with st.form("normal_form"):
            name = st.text_input("姓名")
            age = st.number_input("年龄", min_value=0)
            submitted = st.form_submit_button("提交")
            if submitted:
                st.success(f"提交成功: {name}, {age}岁")

    # 测试2: Tab中的表单
    tab1, tab2 = st.tabs(["Tab1", "Tab2"])

    with tab1:
        with st.form("tab1_form"):
            st.write("Tab1中的表单")
            color = st.selectbox("颜色", ["红", "绿", "蓝"])
            submitted = st.form_submit_button("提交")
            if submitted:
                st.success(f"Tab1提交成功: {color}")

    with tab2:
        with st.form("tab2_form"):
            st.write("Tab2中的表单")
            fruit = st.radio("水果", ["苹果", "香蕉", "橙子"])
            submitted = st.form_submit_button("提交")
            if submitted:
                st.success(f"Tab2提交成功: {fruit}")

    # 测试3: 表单在Tab外定义但在Tab内使用
    with st.form("shared_form"):
        animal = st.selectbox("动物", ["猫", "狗", "鸟"])
        submitted = st.form_submit_button("提交")

    with tab1:
        if submitted:
            st.info(f"共享表单提交: {animal} (在Tab1显示)")


if __name__ == "__main__":
    test_form_in_tab()

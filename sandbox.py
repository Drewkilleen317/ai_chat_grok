import streamlit as st

def call_tab1_function():
    st.write("Executing Tab 1 function")
    st.image("https://www.streamlit.io/images/brand/streamlit-logo-secondary-color.svg", width=200)

def call_tab2_function():
    st.write("Executing Tab 2 function")
    st.write("More content for tab 2.")

def call_tab3_function():
    st.write("Executing Tab 3 function")
    st.line_chart({"x": [1, 2, 3], "y": [4, 5, 6]})

# Create the tabs
tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])  # Give tabs names

# Dictionary mapping tab objects to functions
process_tab_selection = {
    tab1: call_tab1_function,
    tab2: call_tab2_function,
    tab3: call_tab3_function,
}

selected_tab = st.session_state.get('active_tab', chat_tab)
# Determine which tab is active (Streamlit does this implicitly)
for tab, func in process_tab_selection.items():
    if tab:  # This condition is True for the active tab
        func()
        break #Stop once the function is called
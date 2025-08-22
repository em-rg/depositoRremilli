import streamlit as st
if "counter" not in st.session_state:
    st.session_state.counter = 0

# Funzione per aggiornare il contatore
def update_counter(increment_value):
    st.session_state.counter += increment_value


col1, col2, col3 = st.columns(3)

with col1:
    st.button("-", on_click=update_counter, args=(-1,))

with col2:
    st.header(st.session_state.counter)

with col3:
    st.button("+", on_click=update_counter, args=(1,))

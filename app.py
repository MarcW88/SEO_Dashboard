import streamlit as st
import streamlit.components.v1 as components
import os

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dashboard — Semactic · Equans",
    page_icon="📊",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════
# PASSWORD PROTECTION
# ══════════════════════════════════════════════════════════════
def check_password():
    """Returns True if the user has entered the correct password."""
    def password_entered():
        if st.session_state["password"] == "equans2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("<div style='max-width:400px;margin:auto;padding-top:4rem;text-align:center'>", unsafe_allow_html=True)
        st.markdown("### 🔒 Semactic Dashboard")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.markdown("</div>", unsafe_allow_html=True)
        return False
    elif not st.session_state["password_correct"]:
        st.markdown("<div style='max-width:400px;margin:auto;padding-top:4rem;text-align:center'>", unsafe_allow_html=True)
        st.markdown("### 🔒 Semactic Dashboard")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("Incorrect password")
        st.markdown("</div>", unsafe_allow_html=True)
        return False
    return True

if not check_password():
    st.stop()

# ══════════════════════════════════════════════════════════════
# EMBED ORIGINAL DASHBOARD HTML
# ══════════════════════════════════════════════════════════════
# Hide Streamlit chrome for a clean embed
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header {visibility: hidden;}
    .block-container {padding: 0 !important; max-width: 100% !important;}
    [data-testid="stAppViewBlockContainer"] {padding: 0 !important;}
</style>
""", unsafe_allow_html=True)

# Load and render the original dashboard
html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
with open(html_path, "r", encoding="utf-8") as f:
    html_content = f.read()

components.html(html_content, height=5500, scrolling=True)

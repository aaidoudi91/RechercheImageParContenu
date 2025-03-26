import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="PixMatcher App", layout="wide")
st.title("Bienvenue sur l'application PixMatcher")
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Menu de navigation dans la barre latérale
with st.sidebar:
    page = option_menu(
        menu_title="Navigation",
        options=["Recherche", "Visualisation", "À propos"],
        icons=["search", "bar-chart", "info-circle"],
        menu_icon="list",
        default_index=0,
    )
# Charger et exécuter la page sélectionnée correctement
if page == "Recherche":
    import recherche
    recherche.main()  # Appelle la fonction main() correctement
elif page == "Visualisation":
    import visualisation
    visualisation.main()
elif page == "À propos":
    import a_propos
    a_propos.main()
# Exécuter la page sélectionnée
#main()

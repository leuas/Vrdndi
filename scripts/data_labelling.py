'''
run streamlit website


NOTE: run this script in **terminal**
 '''

from src.streamlit.data_labeler import StreamlitDataLabel


if __name__ =='__main__':
    
    st=StreamlitDataLabel()

    st.start_to_lable_data()


# Importing libraries.
import streamlit as st
import codecs
from activities.activity import init_sidebar_content, eda, vis, ml, pandas_profiling
from models.ml_models_datasets import MLDataset

# Main method.
def main():
    '''
    Description: 
            Main method where all the functionalities of the app resides.
            
    Parameters: 
            None
    
    Returns: 
            Nothing
    '''
    title = codecs.open('src//markdowns//title.md', 'r', 'utf-8')
    st.markdown(title.read(), unsafe_allow_html=True)
   
    navigation = init_sidebar_content()

    if navigation == 'Home':
        homepage = codecs.open('src//markdowns//homepage.md', 'r', 'utf-8')
        st.write('\n\n')
        st.markdown(homepage.read(), unsafe_allow_html=True)
    elif navigation == 'Activity Mode':
        activity_mode = st.sidebar.selectbox("Choose an activity", ["Exploratory Data Analysis", "Visualization", "Machine Learning Models", 'Fully Automatic Pandas Profiling'])
        st.write('\n\n')
        
        # Dataset selection.
        dataset_name = st.selectbox("Pick a dataset", ["Iris", "Breast Cancer", "Wine Quality", "Mnist Digits", "Boston Houses", 'Diabetes'])
        dataset = MLDataset(dataset_name)
        df = dataset.get_dataframe()  

        if activity_mode == 'Exploratory Data Analysis':
            eda(df)
        if activity_mode == 'Visualization':
            vis(df)
        if activity_mode == 'Machine Learning Models':
            ml(df)
        if activity_mode == 'Fully Automatic Pandas Profiling':
            pandas_profiling(df)
    elif navigation == 'About':
        aboutpage = codecs.open('src//markdowns//aboutpage.md', 'r', 'utf-8')
        st.write('\n\n')
        st.markdown(aboutpage.read(), unsafe_allow_html=True)
        st.write('\n\n')
        if st.button('Thank You For Using MLify!'):
            st.balloons()
    
            
# Application starts executing here.                
if __name__ == "__main__":
    main()

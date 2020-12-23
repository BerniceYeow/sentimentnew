# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:47:10 2020

@author: BerniceYeow
"""


import pandas as pd


import malaya





import streamlit as st


from PIL import Image

preprocessing = malaya.preprocessing.preprocessing()



def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.title("HATI.AI")
    image = Image.open('macroview.jpg')
    #st.image(image, use_column_width=False)
    st.sidebar.image(image)
    st.sidebar.title("Hati.Ai Web App")
    


            
    @st.cache(suppress_st_warning=True)
    def load_data(uploaded_file):
        

        df = pd.read_csv(uploaded_file)
                
 
        return df
    
    st.sidebar.subheader("Choose What Do You Want To Do")
    classifier = st.sidebar.selectbox(" ", ( "POWER BI Dashboard", "Interact with our chatbot"))
    if classifier == 'POWER BI Dashboard':
        import streamlit.components.v1 as components
        from urllib.request import urlopen
        html = urlopen("https://app.powerbi.com/view?r=eyJrIjoiZTA4NWU4MjYtOTk3Yi00N2ZhLTgwZWQtZWFhMzNkNDk1Zjk3IiwidCI6Ijk5NmQwYTI3LWUwOGQtNDU1Ny05OWJlLTY3ZmQ2Yjk3OTA0NCIsImMiOjEwfQ%3D%3D&pageName=ReportSection06db5928b6af61b2868f").read()
        #components.html(html, width=None, height=600, scrolling=True)
        st.markdown("""
            <iframe width="900" height="606" src="https://app.powerbi.com/view?r=eyJrIjoiZTA4NWU4MjYtOTk3Yi00N2ZhLTgwZWQtZWFhMzNkNDk1Zjk3IiwidCI6Ijk5NmQwYTI3LWUwOGQtNDU1Ny05OWJlLTY3ZmQ2Yjk3OTA0NCIsImMiOjEwfQ%3D%3D&pageName=ReportSection06db5928b6af61b2868f" frameborder="0" style="border:0" allowfullscreen></iframe>
            """, unsafe_allow_html=True)

  
    if classifier == 'Interact with our chatbot':    
        import pickle
        with open('tnb_topic_classifier_svm', 'rb') as training_model:
            topic_model = pickle.load(training_model)

        model = malaya.sentiment.transformer(model = 'tiny-albert')
        #from src import model          
        #malay_bert = model.BertModel()
        # eng_flair = model.Flair()
        # eng_vader = model.Vader()
        test = pd.DataFrame()
        test['Positive'] = ''
        test['Neutral'] = ''
        test['Negative'] = ''
        
        st.title("Sentiment Analyzer")
        message = st.text_area("Enter Text","Type Here ..")
        if st.button("Analyze"):
         with st.spinner("Analyzing the text …"):
             result = model.predict_proba([message])
             #result = malay_bert.predict(message)
             message = [message]
             topic = topic_model.predict(message)
             #output = "Result is: Positive:" + str(result[0]) + "Neutral:" + str(result[1]) + "Negative:" + str(result[2]) + "topic is: " + str(topic)
             output = "result is:" + str(result) + "topic is: " + str(topic)
             st.write(output)

        else:
         st.warning("Not sure! Try to add some more words")






if __name__ == '__main__':
    main()
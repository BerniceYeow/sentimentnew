# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:47:10 2020

@author: BerniceYeow
"""


import pandas as pd


import malaya





import streamlit as st


from PIL import Image




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
    classifier = st.sidebar.selectbox(" ", ("Find new topics automatically", "POWER BI Dashboard", "Interact with our chatbot"))
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


    if classifier == 'Find new topics automatically':

        
        uploaded_file = st.file_uploader('Upload CSV file to begin', type='csv')
    
        #if upload then show left bar
        if uploaded_file is not None:
            df = load_data(uploaded_file)
    
    
    
            if st.sidebar.checkbox("Show raw data", False):
                st.subheader("Uploaded Data Set")
                st.write(df)
    
    

            st.sidebar.subheader("Text column to analyse")
            st_ms = st.sidebar.selectbox("Select Text Columns To Analyse", (df.columns.tolist()))
            

            df_list = list(df)
 

            import top2vec
            from top2vec import Top2Vec
            
            #INITIALIZE AN EMPTY DATAFRAME, CONVERT THE TEXT INTO STRING AND APPEND INTO THE NEW COLUMN
            d1 = pd.DataFrame()
            d1['text'] = ""
            d1['text'] = df[st_ms]
            d1['text'] = d1['text'].astype(str)
            
    
            #INITIALIZE THE TOP2VEC MODEL AND FIT THE TEXT
            #model.build_vocab(df_list, update=False)
            model = Top2Vec(documents=d1['text'], speed="learn", workers=10)
            
            topic_sizes, topic_nums = model.get_topic_sizes()
            for topic in topic_nums:
                st.pyplot(model.generate_topic_wordcloud(topic))
                # Display the generated image:

        




if __name__ == '__main__':
    main()
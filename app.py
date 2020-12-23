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
#IMPORT ALL THE PACKAGES

# Core Pkgs 
import streamlit as st 
import os
import pandas as pd

# NLP Pkgs
from textblob import TextBlob 
import spacy
from gensim.summarization import summarize

# Sentiment Analysis Pkg
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


#-------------------------------------------------------
#FUNCTIONS USED IN THE WEBSITE

# Function for Tokenizer
def text_analyzer(my_text):
    nlp = spacy.load("en_core_web_sm")
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
    return allData

# Function for entity 
def entity_analyzer(my_text):
    nlp = spacy.load("en_core_web_sm")
    docx = nlp(my_text)
    tokens = [ token.text for token in docx]
    entities = [(entity.text,entity.label_)for entity in docx.ents]
    allData = ['"Token":{},\n"Entities":{}'.format(tokens,entities)]
    return allData

# Function for Summarization
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


# Function for Sentiment analysis 
def convert_to_df(sentiment):
	sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
	sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
	return sentiment_df

def analyze_token_sentiment(docx):
	analyzer = SentimentIntensityAnalyzer()
	pos_list = []
	neg_list = []
	neu_list = []
	for i in docx.split():
		res = analyzer.polarity_scores(i)['compound']
		if res > 0.1:
			pos_list.append(i)
			pos_list.append(res)

		elif res <= -0.1:
			neg_list.append(i)
			neg_list.append(res)
		else:
			neu_list.append(i)

	result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
	return result 



#-------------------------------------------------------
#MAIN WEBSITE DESIGN

def main():
    """ NLP Based App with Streamlit """
    st.title("NLP analysis with Streamlit")
    st.subheader("Natural Languages Processing on the go \n (One option at a time)")


    #Tokenization 
    if st.checkbox("Show Tokens and Lemma", key="1"):
        st.subheader("Tokenize your text")
        message1 = st.text_area("Enter your text", "Type here")
        if st.button("Analyze"):
            nlp_result = text_analyzer(message1)
            st.json(nlp_result)
         
    #Named Entity
    if st.checkbox("Show Named Entity",  key="2"):
        st.subheader("Extract entitites from your text")
        message2 = st.text_area("Enter your text", "Type here")
        if st.button("Extract"):
            nlp_result = entity_analyzer(message2)
            st.json(nlp_result)


    #Text Summarization 
    if st.checkbox("Show Text Summarization", key="3"):
        st.subheader("Summary of your text")
        message3 = st.text_area("Enter your text", "Type here")
        summary_options = st.selectbox("Choose Summarizer",['sumy','gensim'])
        if st.button("Summarize"):
            if summary_options == 'sumy':
                st.text("Using Sumy Summarizer ..")
                summary_result = sumy_summarizer(message3)
            elif summary_options == 'gensim':
                st.text("Using Gensim Summarizer ..")
                summary_result = summarize(message3)
            else:
                st.warning("Using Default Summarizer")
                st.text("Using Gensim Summarizer ..")
                summary_result = summarize(message3)
            st.success(summary_result)

    #Sentiment Analysis
    if st.checkbox("Show Sentiment Analysis", key="4"):
        st.subheader("Sentiment of your text")
        with st.form(key='nlpForm'):
            message4 = st.text_area("Enter your text", "Type here")
            submit_button = st.form_submit_button(label='Analyze')

		# layout
        col1,col2 = st.columns(2)
        if submit_button:
            
            with col1:
                st.info("Results")
                sentiment = TextBlob(message4).sentiment
                st.write(sentiment)
                
                # Emoji
                if sentiment.polarity > 0:
                    st.markdown("Sentiment:: Positive :smiley: ")
                elif sentiment.polarity < 0:
                    st.markdown("Sentiment:: Negative :angry: ")
                else:
                    st.markdown("Sentiment:: Neutral 😐 ")
                    
                # Dataframe
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                # Visualization
                c = alt.Chart(result_df).mark_bar().encode(
					x='metric',
					y='value',
					color='metric')
                st.altair_chart(c,use_container_width=True)


            with col2:
                st.info("Token Sentiment")
                
                token_sentiments = analyze_token_sentiment(message4)
                st.write(token_sentiments)


    
    st.sidebar.subheader("About App")
    st.sidebar.text("NLPAnalysis with Streamlit")
    st.sidebar.info("Cudos to the Streamlit Team")
    
    st.sidebar.subheader("By:")
    st.sidebar.text("Lukas Tuong")
    st.sidebar.text("tuong.dong@capgemini.com")


    st.sidebar.subheader("References: ")
    st.sidebar.text("Jesse E.Agbe(JCharis)")


if __name__ == '__main__':
	main()
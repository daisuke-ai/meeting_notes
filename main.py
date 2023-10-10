import os
import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage
import time
from pydub import AudioSegment
from apikey import OPENAI_API_KEY


os.environ['openai_api_key'] = OPENAI_API_KEY

# Function to transcribe an audio file using OpenAI's Whisper API
def transcribe_audio(file_path):
    openai.api_key = openai_api_key

    audio_file = open(file_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    return transcript['text']


# Home page
if 'page' not in st.session_state:
    st.session_state.page = 'home'


def clear_page():
    for _ in range(100):
        st.empty()


button_style = """
<style>
.markdown-text-container button {
    border: 2px solid #008CBA;
    border-radius: 8px;
    background-color: white;
    color: black;
    padding: 14px 28px;
    font-size: 16px;
    cursor: pointer;
    text-align: center;
}
.markdown-text-container button:hover {
    background-color: #008CBA;
    color: white;
}
</style>
"""

st.markdown(button_style, unsafe_allow_html=True)

if st.session_state.page == 'home':
    st.title('🎙️ Meeting Notes Summarizer')
    html_code = """
    <div style="text-align:center;">
        <h1>Welcome to the Meeting Notes Summarizer!</h1>
        <p>Upload your meeting audio and get a summarized version of the notes.</p>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

    if st.button('Get Started'):
        st.session_state.page = 'input_page'
        st.experimental_rerun()

elif st.session_state.page == 'input_page':
    clear_page()
    st.title('Upload Your Meeting Audio')

    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        with open("temp_audio.mp3", "wb") as f:
            f.write(uploaded_file.getbuffer())

        transcription = transcribe_audio("temp_audio.mp3")
        st.write("Transcription:", transcription)



        def summarize_notes(prompt):
            llm = ChatOpenAI(model_name="gpt-4", temperature=0.9)
            prompt_template = PromptTemplate.from_template(prompt)
            notes_chain = LLMChain(llm=llm, prompt=prompt_template)
            response_dict = notes_chain({'prompt': prompt})
            return response_dict['text']


        initial_prompt = f"Summarize the following meeting notes: {transcription}"

        generate_button = st.button('Generate Summary')

        if generate_button:
            with st.spinner('Generating your summary...'):
                summary = summarize_notes(initial_prompt)
                time.sleep(2)
            st.success('Summary generated!')
            st.write(summary)

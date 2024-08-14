import textwrap
import os
import getpass
import google.generativeai as genai
import PIL.Image
import pandas as pd

# Setup API key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")
GOOGLE_API_KEY =  os.environ["GOOGLE_API_KEY"]

chat_model_name = 'gemini-1.5-flash'
embedding_model_name = 'models/embedding-001'

# Steps
# 1) basic chatbot
# 2) multimodal chatbot
# 3) chatbot with memory
# 4) Build vector database


# TODO check this link for multimodal embeddings perhaps.
# https://medium.com/kx-systems/guide-to-multimodal-rag-for-images-and-text-10dab36e3117

def main():
    # chatbot()
    # multimodal_chatbot()
    # chatbot_with_memory()
    build_vectorbase()

def build_vectorbase():
    DOCUMENT1 = {
    "title": "Operating the Climate Control System",
    "content": "Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console.  Temperature: The temperature knob controls the temperature inside the car. Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. Airflow: The airflow knob controls the amount of airflow inside the car. Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. Mode: The mode button allows you to select the desired mode. The available modes are: Auto: The car will automatically adjust the temperature and airflow to maintain a comfortable level. Cool: The car will blow cool air into the car. Heat: The car will blow warm air into the car. Defrost: The car will blow warm air onto the windshield to defrost it."}
    DOCUMENT2 = {
        "title": "Touchscreen",
        "content": "Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon.  For example, you can touch the \"Navigation\" icon to get directions to your destination or touch the \"Music\" icon to play your favorite songs."}
    DOCUMENT3 = {
        "title": "Shifting Gears",
        "content": "Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions."}

    documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]

    df = pd.DataFrame(documents)
    df.columns = ['Title', 'Text']

    df['Embeddings'] = df.apply(lambda row: embed_fn(row['Title'], row['Text']), axis=1)
    print(df)

def embed_fn(title, text):
  return genai.embed_content(model=embedding_model_name,
                             content=text,
                             task_type="retrieval_document",
                             title=title)["embedding"]


def chatbot_with_memory():
    system_instruction="You are an AI assistant. Answer my questions concisely"
    model = genai.GenerativeModel(model_name=chat_model_name, system_instruction=system_instruction)
    chat = model.start_chat(history=[])
    
    response = chat.send_message("In one sentence, explain how a computer works to a young child.")
    print('\n', response.text)
    print(chat.history)

    response = chat.send_message("Okay, how about a more detailed explanation to a high schooler?")
    print('\n', response.text)
    print(chat.history)

def multimodal_chatbot():
    system_instruction="You are an AI assistant. Answer my questions concisely"
    model = genai.GenerativeModel(model_name=chat_model_name, system_instruction=system_instruction)

    image = PIL.Image.open('../data/images/image.png')
    response = model.generate_content(["Write a short, engaging blog post based on this picture. It should include a description of the meal in the photo and talk about my journey meal prepping.", image])
    print('\n', response.text)

def chatbot():
    system_instruction="You are an AI assistant. Answer my questions concisely"
    model = genai.GenerativeModel(model_name=chat_model_name, system_instruction=system_instruction)
    response = model.generate_content("What is the meaning of life?")
    print('\n' + response.text)

if __name__ == '__main__':
    main()

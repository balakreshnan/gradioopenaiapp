from io import BytesIO
import gradio as gr
import openai
from PyPDF2 import PdfReader
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import base64
import PIL

load_dotenv()


#llm = OpenAI(temperature=0)
openai.api_type = "azure"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version = "2023-09-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

# def abstract_extract(uploaded_file):
#     pdf_bytes = BytesIO(uploaded_file)
#     pdf_reader = PdfReader(pdf_bytes)
    
#     abstract = ""
    
#     for page_number in range(len(pdf_reader.pages)):
#         text = pdf_reader.pages[page_number].extract_text()
        
#         if "abstract" in text.lower():
#             start_index = text.lower().find("abstract")  
#             end_index = text.lower().find("introduction")
#             abstract = text[start_index:end_index]
#             break
    
#     return abstract

def summarize_and_speech(image, text1):
    #abstract_text = abstract_extract(pdf_file)

    #print(abstract_text)
    #try:
    #    summary = summarization(abstract_text, max_length=15, min_length=10)
    #except:
    #    pass

    #summary = summarization(abstract_text, max_length=15, min_length=10)

    #try:
        #imagetext = imageprocess(image, text1)
    #except:
    #    pass

    #print(image.shape)
    imagetext = imageprocess(image, text1)

    #tts_output = synthesiser(summary)
    #audio_data = tts_output[0]["audio"]

    return imagetext #summary #, imagetext #, audio_data


def imageprocess(image, text1):
    #encoded_image = base64.b64encode(image).decode('utf-8')

    im_file = BytesIO()
    image.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes)
    im_b64 = im_b64.decode("utf-8")
    encoded_image = im_b64

    #encoded_image = image
    message_text = [
        {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are an AI assistant that helps people find information."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{encoded_image}"
          }
        },
        {
          "type": "text",
          "text": text1
        }
      ]
    }
    ]
    #print(encoded_image)

    client = AzureOpenAI(
    azure_endpoint = os.getenv('OPENAI_API_BASE'), 
    api_key=os.getenv('OPENAI_API_KEY'),
    api_version="2023-09-01-preview"
    )

    response = client.chat.completions.create(
        model="gpt-4-vision", # model = "deployment_name". Ggpt4, gpt4 version 1106
        messages=message_text
    )


    return response.choices[0].message.content


iface = gr.Interface(
    fn=summarize_and_speech,
    inputs=[gr.Image(type="pil", label="Upload Research Paper PDF File"),
            #"image",
            gr.Textbox(label="Text 1", info="Initial text", lines=3, value="Describe the image?",), 
            ],
    outputs=[gr.Textbox(label="Abstract Summary:")],
    live=True,
    title="Abstract Research Paper Summarizer",
    description="Upload a Research Paper PDF File. The model will generate a one line summary of the Abstract section and a speech audio."
)

if __name__ == "__main__":
    iface.launch()
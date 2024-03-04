import os
from openai import AzureOpenAI
import gradio as gr
from dotenv import dotenv_values
import time

config = dotenv_values("env.env")

client = AzureOpenAI(
  azure_endpoint = config["AZURE_OPENAI_ENDPOINT"], 
  api_key=config["AZURE_OPENAI_KEY"],  
  api_version="2023-09-01-preview"
)

def predict(message, history, firstllm, system_prompt, prompt_1):
    history_openai_format = []
    history_openai_format.append({"role": "system", "content": system_prompt })
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})

    start_time = time.time()

    print('model:', firstllm)

    response = client.chat.completions.create(
        #model= "gpt-35-turbo", #"gpt-4-turbo", # model = "deployment_name".
        model=firstllm,
        messages=history_openai_format,
        #stream=True
    )

    partial_message = ""
    # calculate the time it took to receive the response
    response_time = time.time() - start_time

    #clsprompt_1 = response.choices[0].message.content
    #gr.Textbox.update(label=f"Question {response.choices[0].message.content}", value=f"Question {response.choices[0].message.content}")
    #text1.update(value=f"Question {response.choices[0].message.content}")

    # print the time delay and text received
    print(f"Full response from model {firstllm} received {response_time:.2f} seconds after request")
    #print(f"Full response received:\n{response}")

    returntext = response.choices[0].message.content + f" \nTime Taken: ({response_time:.2f} seconds)"

    #return response.choices[0].message.content + f" \nTime Taken: ({response_time:.2f} seconds)"
    return returntext

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            text1 = gr.Textbox(label="prompt 1")
            inbtw = gr.Button("Submit")
        with gr.Column(scale=2, min_width=600):
            chatbot = gr.ChatInterface(predict, title="LLM Model", additional_inputs=[gr.Dropdown(
                ["gpt-4-turbo", "gpt-35-turbo", "llama27b"], label="firstllm"
            ),gr.Textbox("You are helpful AI.", label="System Prompt"),])

if __name__ == "__main__":
    demo.launch()
from openai import OpenAI
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

client = OpenAI()


def rephrase_text(text, model="gpt-3.5-turbo"):
    """
    Rephrases the given text using OpenAI's GPT model.

    :param text: The text to be rephrased.
    :param model: The model to be used for rephrasing. Default is "text-davinci-003".
    :return: The rephrased text.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"Paraphrase the following text: {text}"}
        ]
    )
    return response.choices[0].message.content


def add_to_file(file_name, string_to_add):
    with open(file_name, 'a') as file:  # 'a' mode opens the file for appending
        file.write(string_to_add + "\n")  # Appends the string and a newline to the file


original_text = "Climate change is one of the most significant challenges facing humanity."
paraphrased_text = rephrase_text(original_text)
print("Original:", original_text)
print("Rephrased:", paraphrased_text)

output_filename = "output.txt"

add_to_file(output_filename, paraphrased_text)

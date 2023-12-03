from openai import OpenAI
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


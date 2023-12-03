from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
client = OpenAI()


def gpt_parser(sentence, model="gpt-3.5-turbo", temperature=0.0):
    """
    Parses the input sentence to AMR using OpenAI GPT API.

    :return: A parsed structure of the input sentence in AMR format in the form of a dictionary.
    """
    # Define the delimiter to be used for the system message and the user query
    delimiter = "####"

    # Set up the system message to be sent to messages to the model
    system_message = f"""
    Parse the user input sentence to AMR, and then convert it into Python dictionary.
    The user query will be delimited with four hashtags,\
    i.e. {delimiter}.

    For example,
    Sentence: Consequently , experts have predicted that the world has at least ten years within which to lower the volume of greenhouse gas emissions before it is too late to redeem the situation .
    Parsed AMR: (c / cause-01
      :ARG1 (p / predict-01
            :ARG0 (p2 / person
                  :ARG1-of (e / expert-01))
            :ARG1 (h / have-03
                  :ARG0 (w / world)
                  :ARG1 (y / year
                        :quant (a / at-least :op1 10)
                        :purpose (l / lower-05
                              :ARG0 w
                              :ARG1 (v / volume
                                    :mod (e2 / emit-01
                                          :ARG0 w
                                          :ARG1 (g / gas
                                                :mod (g2 / greenhouse))))
                              :time (b / before
                                    :op1 (h2 / have-degree-91
                                          :ARG2 (l2 / late)
                                          :ARG3 (t / too)
                                          :ARG6 (r / redeem-01
                                                :ARG1 (s / situation)))))))))
    """

    # Set up the user message to be sent to messages to the model
    user_message = f"""
    This is the user input sentence: {sentence}
    """

    # Set up the messages to be sent to the model, including the system message and the user query
    messages = [
        {'role': 'system',
         'content': system_message},
        {'role': 'user',
         'content': f"{delimiter}{user_message}{delimiter}"},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


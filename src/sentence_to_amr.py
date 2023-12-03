from parsers import *
sentence_1 = "Climate change is one of the most significant challenges facing humanity."
print(sentence_1)
sentence_1_amr = gpt_parser(sentence_1)
print(sentence_1_amr)


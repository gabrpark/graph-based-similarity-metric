import os


def read_sentences_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        with open(file_path, 'r') as file:
            sentences = file.readlines()
            print(sentences)
        return [sentence.strip() for sentence in sentences]


def add_to_file(file_path, string_to_add):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        with open(file_path, 'a') as file:  # 'a' mode opens the file for appending
            file.write(string_to_add + "\n")  # Appends the string and a newline to the file

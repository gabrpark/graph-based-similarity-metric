from rephraser import rephrase_text
from utils import read_sentences_from_file, add_to_file
from sentence_transformers import SentenceTransformer
from pathlib import Path


def main():
    # Get the current working directory as a Path object
    current_directory_path = Path.cwd()
    output_filepath = current_directory_path / "data/output.txt"

    original_sentence = "Climate change is one of the most significant challenges facing humanity."
    rephrased_sentence = "Climate change poses a major threat to mankind and stands as one of the most prominent obstacles we confront."
    print("Original:", original_sentence)
    print("Rephrased:", rephrased_sentence)

    add_to_file(output_filepath, rephrased_sentence)

    # Paths to your files
    original_text_filepath = current_directory_path / "data/input.txt"
    rephrased_text_filepath = current_directory_path / "data/output.txt"

    # Read sentences from files
    original_sentences = read_sentences_from_file(original_text_filepath)
    rephrased_sentences = read_sentences_from_file(rephrased_text_filepath)


if __name__ == "__main__":
    main()

from metrics import *
from utils import add_to_file
from pathlib import Path


reference_sentence = "The cat is on the mat."
candidate_sentence = "A cat is sitting on the mat."

# Calculating the BLEU score
bleu_score = calculate_bleu(reference_sentence, candidate_sentence)
print(f"BLEU score: {bleu_score}")

# Calculating Precision, Recall, and F1 scores using BERTScore
bertscore_precision, bertscore_recall, bertscore_f1 = calculate_bertscore(reference_sentence, candidate_sentence)
print(f"BERTScore: Precision - {bertscore_precision}, Recall - {bertscore_recall}, F1 - {bertscore_f1}")

# Calculating ROUGE scores
rouge_scores = calculate_rouge(candidate_sentence, reference_sentence)
print("ROUGE scores:", rouge_scores)

# Calculating the METEOR score
meteor_score = calculate_meteor(candidate_sentence, reference_sentence)
print("METEOR score:", meteor_score)

# Calculating the TF-IDF Cosine Similarity score
tfidf_cosine_sim_score = calculate_cosine_similarity(candidate_sentence, reference_sentence)
print("TF-IDF Cosine Similarity:", tfidf_cosine_sim_score)


# Calculating the Semantic Textual Similarity score
similarity = semantic_textual_similarity("sentence one", "sentence two")
print("Semantic Textual Similarity:", similarity)


# Add all the scores to the metrics_output.txt file
current_directory_path = Path.cwd()
metrics_output_filepath = current_directory_path / "../data/metrics_output.txt"

add_to_file(metrics_output_filepath, f"BLEU score: {bleu_score}")
add_to_file(metrics_output_filepath, f"BERTScore: Precision - {bertscore_precision}, Recall - {bertscore_recall}, F1 - {bertscore_f1}")
add_to_file(metrics_output_filepath, f"ROUGE scores: {rouge_scores}")
add_to_file(metrics_output_filepath, f"METEOR score: {meteor_score}")
add_to_file(metrics_output_filepath, f"TF-IDF Cosine Similarity: {tfidf_cosine_sim_score}")
add_to_file(metrics_output_filepath, f"Semantic Textual Similarity: {similarity}")
add_to_file(metrics_output_filepath, "\n")
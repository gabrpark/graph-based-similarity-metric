from nltk.translate.bleu_score import sentence_bleu
from bert_score import score
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


def calculate_bleu(reference_sentence, candidate_sentence):
    """
    Calculate the BLEU score for a pair of sentences.

    This function computes the BLEU score, a measure of similarity between a candidate sentence and a reference sentence.

    Parameters:
    reference_sentence (str): The reference sentence against which the candidate is evaluated.
    candidate_sentence (str): The candidate sentence to be evaluated.

    Returns:
    float: The BLEU score for the given sentences.
    """
    # Splitting the reference sentence into a list of words wrapped in another list (required by sentence_bleu)
    reference = [reference_sentence.split()]
    # Splitting the candidate sentence into a list of words
    candidate = candidate_sentence.split()
    # Calculating the BLEU score
    score = sentence_bleu(reference, candidate)
    return score


def calculate_bertscore(reference_sentence, candidate_sentence):
    """
    Calculate the BERTScore for a pair of sentences.

    BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference sentences by cosine similarity.

    Parameters:
    reference_sentence (str): The reference sentence.
    candidate_sentence (str): The candidate sentence.

    Returns:
    tuple: A tuple containing the Precision, Recall, and F1 score.
    """
    # Calculating Precision, Recall, and F1 scores using BERTScore
    P, R, F1 = score([candidate_sentence], [reference_sentence], lang="en")
    return P.mean(), R.mean(), F1.mean()


def calculate_rouge(candidate_sentence, reference_sentence):
    """
    Calculate the ROUGE score for a pair of sentences.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics for evaluating automatic summarization and machine translation.

    Parameters:
    candidate_sentence (str): The candidate sentence or summary.
    reference_sentence (str): The reference sentence or summary.

    Returns:
    dict: A dictionary of ROUGE scores.
    """
    rouge = Rouge()
    # Calculating ROUGE scores
    scores = rouge.get_scores(candidate_sentence, reference_sentence)
    return scores


def calculate_meteor(candidate_sentence, reference_sentence):
    """
    Calculate the METEOR score for a pair of sentences.

    METEOR (Metric for Evaluation of Translation with Explicit Ordering) is a metric for the evaluation of machine translation output.

    Parameters:
    candidate_sentence (str): The candidate sentence.
    reference_sentence (str): The reference sentence.

    Returns:
    float: The METEOR score.
    """

    # Tokenize the sentences
    tokenized_reference = reference_sentence.split()
    tokenized_candidate = candidate_sentence.split()

    # Calculating the METEOR score
    score = meteor_score([tokenized_reference], tokenized_candidate)
    return score


def calculate_cosine_similarity(text1, text2):
    """
    Calculate the cosine similarity between two texts.

    This function uses TF-IDF to convert the texts into vector space and computes the cosine similarity between these vectors.

    Parameters:
    text1 (str): The first text.
    text2 (str): The second text.

    Returns:
    float: The cosine similarity score between the two texts.
    """
    vect = TfidfVectorizer(min_df=1, stop_words="english")
    # Transforming the texts into TF-IDF vectors
    tfidf = vect.fit_transform([text1, text2])
    # Calculating cosine similarity between the two vectors
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]


def semantic_textual_similarity(str1, str2):
    """
    Calculate the semantic textual similarity between two strings using a pre-trained Sentence Transformer model.

    This function utilizes the 'all-MiniLM-L6-v2' model from Sentence Transformers to encode two input strings into
    embeddings and then computes the cosine similarity between these embeddings. The cosine similarity is a measure
    of how similar the two text embeddings are, ranging from -1 (completely different) to 1 (exactly the same).

    Parameters:
    str1 (str): The first input string to compare.
    str2 (str): The second input string to compare.

    Returns:
    float: The cosine similarity between the embeddings of the two input strings, indicating their semantic similarity.
    """

    # Load the pre-trained Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for the first string, converting to tensor for cosine similarity calculation
    embedding1 = model.encode(str1, convert_to_tensor=True)

    # Generate embeddings for the second string
    embedding2 = model.encode(str2, convert_to_tensor=True)

    # Compute and return the cosine similarity between the two embeddings
    return util.pytorch_cos_sim(embedding1, embedding2).item()






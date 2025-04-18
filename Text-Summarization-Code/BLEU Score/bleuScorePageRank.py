from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def calculate_bleu_score(reference_summaries, candidate_summaries):
    """
    Calculate BLEU score for text summarization.
    
    Parameters:
    - reference_summaries: List of lists, where each inner list contains one or more reference summaries.
    - candidate_summaries: List of generated summaries by the model.
    
    Returns:
    - BLEU score.
    """
    # Ensure input format is compatible with NLTK's BLEU score
    formatted_references = [[ref.split()] for ref in reference_summaries]
    formatted_candidates = [cand.split() for cand in candidate_summaries]
    
    # Calculate BLEU score
    smooth_func = SmoothingFunction().method1
    score = corpus_bleu(formatted_references, formatted_candidates, smoothing_function=smooth_func)
    return score

# Example usage
reference_summaries = [
    """ 
    Natural Language Processing (NLP) is a sub-field of artificial intelligence (AI).
    It focuses on the interaction between computers and human language.
    The ultimate goal of NLP is to enable computers to understand, interpret, and generate human languages.
    Applications of NLP include chatbots, sentiment analysis, and language translation.
    Recent advancements in NLP are largely driven by deep learning and large language models like GPT.
    """
]

candidate_summaries = [
    "It focuses on the interaction between computers and human language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human languages."
]

# Compute BLEU score
bleu_score = calculate_bleu_score(reference_summaries, candidate_summaries)
print(f"BLEU score: {bleu_score:.4f}")

from rouge_score import rouge_scorer

def calculate_rouge(reference, generated):
    """
    Calculate ROUGE scores between a reference summary and a generated summary.
    """
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores

# Example usage
if __name__ == "__main__":
    reference_summary = (
    """ 
    Natural Language Processing (NLP) is a sub-field of artificial intelligence (AI).
    It focuses on the interaction between computers and human language.
    The ultimate goal of NLP is to enable computers to understand, interpret, and generate human languages.
    Applications of NLP include chatbots, sentiment analysis, and language translation.
    Recent advancements in NLP are largely driven by deep learning and large language models like GPT.
    """
    )
    generated_summary = (
        "It focuses on the interaction between computers and human language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human languages."
    )

    # Calculate ROUGE scores
    rouge_scores = calculate_rouge(reference_summary, generated_summary)

    print("ROUGE Scores:")
    for key, value in rouge_scores.items():
        print(f"{key}: Precision: {value.precision:.4f}, Recall: {value.recall:.4f}, F1 Score: {value.fmeasure:.4f}")

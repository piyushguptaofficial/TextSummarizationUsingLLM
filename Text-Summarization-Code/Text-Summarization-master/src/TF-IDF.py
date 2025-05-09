from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import math
import nltk
nltk.download('stopwords')

ps = PorterStemmer()


def sent_preprocessing(sentences: list) -> list:
    cleaned_sentencs = [sent for sent in sentences if sent]
    for sent in sentences:
        if sent == '' or sent == ' ':
            print(1)
    return cleaned_sentencs


def text_preprocessing(sentences: list):
    """
    Pre processing text to remove unnecessary words.
    """
    # print('Preprocessing text')

    stop_words = set(stopwords.words('english'))

    clean_words = None
    for sent in sentences:
        words = word_tokenize(sent)
        words = [ps.stem(word.lower()) for word in words if word.isalnum()]
        clean_words = [word for word in words if word not in stop_words]

    return clean_words


def create_tf_matrix(sentences: list) -> dict:
    """
    Here document refers to a sentence.
    TF(t) = (Number of times the term t appears in a document) / (Total number of terms in the document)
    """
    # print('Creating tf matrix.')

    tf_matrix = {}

    for sentence in sentences:
        tf_table = {}

        clean_words = text_preprocessing([sentence])
        words_count = len(word_tokenize(sentence))

        # Determining frequency of words in the sentence
        word_freq = {}
        for word in clean_words:
            word_freq[word] = (word_freq[word] + 1) if word in word_freq else 1

        # Calculating relative tf of the words in the sentence
        for word, count in word_freq.items():
            tf_table[word] = count / words_count

        tf_matrix[sentence[:15]] = tf_table

    return tf_matrix


def create_idf_matrix(sentences: list) -> dict:
    """
    Inverse Document Frequency.
    IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
    """
    # print('Creating idf matrix.')

    idf_matrix = {}
    documents_count = len(sentences)
    sentence_word_table = {}

    # Getting words in the sentence
    for sentence in sentences:
        clean_words = text_preprocessing([sentence])
        sentence_word_table[sentence[:15]] = clean_words

    # Determining word count table with the count of sentences which contains the word.
    word_in_docs = {}
    for sent, words in sentence_word_table.items():
        for word in words:
            word_in_docs[word] = (word_in_docs[word] + 1) if word in word_in_docs else 1

    # Determining idf of the words in the sentence.
    for sent, words in sentence_word_table.items():
        idf_table = {}
        for word in words:
            idf_table[word] = math.log10(documents_count / float(word_in_docs[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def create_tf_idf_matrix(tf_matrix, idf_matrix) -> dict:
    """
    Create a tf-idf matrix which is multiplication of tf * idf individual words
    """
    # print('Calculating tf-idf of sentences.')

    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def create_sentence_score_table(tf_idf_matrix) -> dict:
    """
    Determining average score of words of the sentence with its words tf-idf value.
    """
    # print('Creating sentence score table.')

    sentence_value = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0
        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        smoothing = 1
        sentence_value[sent] = (total_score_per_sentence + smoothing) / (count_words_in_sentence + smoothing)

    return sentence_value


def find_average_score(sentence_value):
    """
    Calculate average value of a sentence form the sentence score table.
    """
    # print('Finding average score')

    sum = 0
    for val in sentence_value:
        sum += sentence_value[val]

    average = sum / len(sentence_value)

    return average


def generate_summary(sentences, sentence_value, threshold):
    """
    Generate a sentence for sentence score greater than average.
    """
    # print('Generating summary')

    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentence_value and sentence_value[sentence[:15]] >= threshold:
            summary += sentence + " "
            sentence_count += 1

    return summary


def main():
    text = ""
    with open('D:/JAVA Projects/Text Summarization/Text-Summarization-master/src/File_1_en', "r+", encoding='UTF8') as f:
        for line in f:
            text += line

    sentences = sent_tokenize(text)
    # print('Sentences', sentences)

    # sentences = sent_preprocessing(sentences)

    tf_matrix = create_tf_matrix(sentences)
    # print('TF matrix', tf_matrix)

    idf_matrix = create_idf_matrix(sentences)
    # print('IDF matrix',idf_matrix)

    tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)
    # print('TF-IDF matrix', tf_idf_matrix)
    # print('First document tfidf',tf_idf_matrix[list(tf_idf_matrix.keys())[0]])

    sentence_value = create_sentence_score_table(tf_idf_matrix)
    # print('Sentence Scores', sentence_value)

    threshold = find_average_score(sentence_value)
    # print('Threshold', threshold)

    summary = generate_summary(sentences, sentence_value, threshold)

    # print('\nOriginal document\n',text,end='\n'*2)
    print('Summary\n', summary)

    print()
    print(f'Original {len(sent_tokenize(text))} sentences, Summarized {len(sent_tokenize(summary))} sentences')


if __name__ == '__main__':
    main()

from lsa_summarizer import LsaSummarizer
import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
import time

start_time = time.time()
source_file = "D:/JAVA Projects/Text Summarization/LSA-Text-Summarization-master/original_text.txt"

with open(source_file, "r", encoding='utf-8') as file:
    text = file.readlines()



summarizer = LsaSummarizer()

stopwords = stopwords.words('portuguese')
summarizer.stop_words = stopwords
summary =summarizer(text[0], 3)

print("====== Original text =====")
print(text)
print("====== End of original text =====")



print("\n========= Summary =========")

print(" ".join(summary))
print("========= End of summary =========")

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

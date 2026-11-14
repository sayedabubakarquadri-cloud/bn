import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
text="Data Visualization is a way to express your data in a visual context so that patterns,correlations,trends betwne the data can be easily undestood"
words=word_tokenize(text)
unigram=list(ngrams(words,1))
print("Unigram:",unigram)
bigram=list(ngrams(words,2))
print("Bigram:",bigram)
trigram=list(ngrams(words,3))
print("Trigram:",trigram)
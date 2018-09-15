import nltk
from nltk.collocations import BigramCollocationFinder

bigram_measures = nltk.collocations.BigramAssocMeasures()
text = "we are the greatest champions of the world. i am the greatest champion of the world"
length = len(text.split())
finder = BigramCollocationFinder.from_words(nltk.word_tokenize(text))
list = finder.nbest(bigram_measures.pmi, length)
print(list)


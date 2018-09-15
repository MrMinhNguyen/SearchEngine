from collections import Counter
from timeit import default_timer as timer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords


# def jaccard(a, b):
#     a = set(a)
#     b = set(b)
#     c = a.intersection(b)
#     return float(len(c)) / (len(a) + len(b) - len(c))

def handleSpecialChar(target):
    temp = ''
    for char in target:
        if char in ['!', '?', '-', '(', ')', ':', '/', '', ',', '#', '"']:
            continue
        if char in ['\n', '^', '*', '@', '.']:
            char = ' '
        temp += char
    return temp


def handleEmpty(target):
    temp = []
    for i in target:
        if i != '':
            temp.append(i)
    return temp

def handleStopWords(target):
    stopWords = stopwords.words('english')
    temp = []
    for i in target:
        if i not in stopWords:
            temp.append(i)
    return temp

def synonym(list, number):
    synonyms = []

    for i in range(len(list)):
        limit = 0
        for syn in wordnet.synsets(list[i]):
            for l in syn.lemmas():
                synonyms.append(l.name())
                if limit == number:
                    break
                limit += 1

    return synonyms

# a jap bike and call myself axis motors tuba irwin
def similarity(synNum, filterNum):
    synList = synonym(searchList, synNum)
    synString = ' '.join(synList)
    newsgroups_train = fetch_20newsgroups(subset='train')

    dict = {}
    dictSyn = {}

    count = 0
    for i in range(len(newsgroups_train.data)):
        line = newsgroups_train.data[i]

        line = handleSpecialChar(line)
        array = line.lower().split(' ')
        array = list(array)
        array = handleEmpty(array)
        array = handleStopWords(array)
        line = ' '.join(array)

        # jac = jaccard(searchList, array)
        # jaccardList.append(jac)

        combine = [searchString, line]
        vectors = TfidfVectorizer().fit_transform(combine)
        array_cos = cosine_similarity(vectors[0], vectors[1])[0][0]
        dict[i] = array_cos

        combineSyn = [synString, line]
        vectors_syn = TfidfVectorizer().fit_transform(combineSyn)
        array_cos_syn = cosine_similarity(vectors_syn[0], vectors_syn[1])[0][0]
        dictSyn[i] = array_cos_syn

        if array_cos > 0:
            count += 1

    for key in list(dict):
        if dict[key] == 0:
            del dict[key]

    for key in list(dictSyn):
        if dictSyn[key] == 0:
            del dictSyn[key]

    resultDict = {}
    for item in dict.keys():
        if item in dictSyn.keys():
            resultDict[item] = dict[item]

    top = Counter(resultDict).most_common(filterNum)
    print(top)


running = True

while (running):
    searchString = input("Give me your string:\n")
    searchList = searchString.split(' ')

    start = timer()
    numSyn = 3
    numFilter = 10
    similarity(numSyn, numFilter)
    end = timer()
    time = end - start
    print(time, " seconds")

    search_again = input("Search again? (Type y or yes to search again, otherwise type anything) \n")
    if not search_again.lower().startswith('y'):
        break

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import wordnet

from nltk.corpus import stopwords
stopWords = stopwords.words('english')

def n_max_elements(list, n, index_list, result_list):
    clone = list[:]

    for i in range(0, n):
        max = 0
        for j in range(len(clone)):
            if clone[j] > max:
                max = clone[j]

        index_list.append(clone.index(max))
        clone.remove(max)
        result_list.append(max)

    print(index_list)
    print(result_list)


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

# list1 = [1,2,3,4,5,6]
# list2 = [3, 5, 7, 9]
# print(list(set(list1).intersection(list2)))

# temp = synonym(["hello", "bye"], 2)
# str = ' '.join(temp)
# print(str, "\n", temp)

searchString = input("Give me your string:\n")
searchList = searchString.split(' ')

synList = synonym(searchList, 3)
synString = ' '.join(synList)

newsgroups_train = fetch_20newsgroups(subset='train')

# jaccardList = []
cosineList = []
cosineSynList = []

# a jap bike and call myself axis motors tuba irwin

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
    cosineList.append(array_cos)

    combineSyn = [synString, line]
    vectors_syn = TfidfVectorizer().fit_transform(combineSyn)
    array_cos_syn = cosine_similarity(vectors_syn[0], vectors_syn[1])[0][0]
    cosineSynList.append(array_cos_syn)

    if array_cos > 0:
        count += 1

index_list = []
result_list = []
# n_max_elements(jaccardList, 5, index_list, result_list)
n_max_elements(cosineList, 10, index_list, result_list)

index_list_syn = []
result_list_syn = []
n_max_elements(cosineSynList, 10, index_list_syn, result_list_syn)

print("Number of similarity = 0: ", count)

print(list(set(index_list).intersection(index_list_syn)))

print(index_list)
print(index_list_syn)

# print("jaccardList", len(jaccardList))
# print("cosineList: ", len(cosineList))
# print("data: ", len(newsgroups_train.data))

# st = handleSpecialChar(newsgroups_train.data[8832])
# lt = st.lower().split(' ')
# lt = list(lt)
# lt = handleEmpty(lt)
# print(lt)



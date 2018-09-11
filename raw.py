from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# import nltk
# from nltk.corpus import wordnet

from nltk.corpus import stopwords
stopWords = stopwords.words('english')

def n_max_elements(list, n, indexes, results):
    final_list = []
    index_list = []

    for i in range(0, n):
        max = 0
        for j in range(len(list)):
            if list[j] > max:
                max = list[j]

        index_list.append(list.index(max))
        list.remove(max)
        final_list.append(max)

    indexes = index_list
    results = final_list
    print(indexes)
    print(results)


def jaccard(a, b):
    a = set(a)
    b = set(b)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


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

# def synonym(list):
#     synonyms = []
#
#     for i in range(len(list)):
#         limit = 0
#         for syn in wordnet.synsets(list[i]):
#             for l in syn.lemmas():
#                 synonyms.append(l.name())
#                 if limit == 3:
#                     break
#                 limit += 1
#
#     return synonyms

# temp = synonym(["hello", "bye"])
# str = ' '.join(temp)
# print(str, temp)

searchString = input("Give me your string:\n")
searchList = searchString.split(' ')

newsgroups_train = fetch_20newsgroups(subset='train')

jaccardList = []
cosineList = []

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

    jac = jaccard(searchList, array)
    jaccardList.append(jac)

    combine = [searchString, line]
    vectors = TfidfVectorizer().fit_transform(combine)
    array_cos = cosine_similarity(vectors[0], vectors[1])[0][0]
    cosineList.append(array_cos)

    if array_cos > 0:
        count += 1

index_list = []
result_list = []
copyJaccardList = jaccardList[:]
copyCosineList = cosineList[:]
n_max_elements(copyJaccardList, 10, index_list, result_list)
n_max_elements(copyCosineList, 10, index_list, result_list)

print(count)
# print("jaccardList", len(jaccardList))
# print("cosineList: ", len(cosineList))
# print("data: ", len(newsgroups_train.data))

# st = handleSpecialChar(newsgroups_train.data[8832])
# lt = st.lower().split(' ')
# lt = list(lt)
# lt = handleEmpty(lt)
# print(lt)



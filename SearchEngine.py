# ---------------------------------------------------------- #
# ---------------------- RMIT Vietnam ---------------------- #
# -------------------- Semester B, 2018 -------------------- #
#
# ----- ISYS2090 - File Structures and Database System ----- #
# ------------- Lecturer: Dr. Vladimir Mariano ------------- #
#
# ---------------------- Assignment 3 ---------------------- #
#
# --------- Author: Nguyen Hoang Minh - s3634696
#                   Vo Quoc Vu - s3575819
#                   Ho Minh Tri - s3594986
#                   Le Viet Hoang Dung - s3568452  --------- #
# ---------------------------------------------------------- #


#
# --------------------------- #
# --- Importing libraries --- #
# --------------------------- #
#
from collections import Counter
from timeit import default_timer as timer
import datetime 
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder

#
# ------------------------------------------------- #
# --- Defining a new class to store the Searces --- #
# ------------------------------------------------- #
#
class Search:
    def __init__(self, keywords):

        # Each search is stored with 2 fields: the content and the time
        self.keywords = keywords
        self.time = datetime.datetime.now()

#
# -------------------------------- #
# --- Defining a new functions --- #
# -------------------------------- #
#

# This function removes and handle special characters in a given string
def handleSpecialChar(target):
    # Create a new empty string to return back later
    temp = ''

    # For each character in the target string
    for char in target:
        # If it's '!', '?', '-', '(', ')', ':', '/', '', ',', '#' or '"' then ignore it
        if char in ['!', '?', '-', '(', ')', ':', '/', '', ',', '#', '"']:
            continue
        # If it's '\n', '^', '*', '@' or '.' then replace it with ' '
        if char in ['\n', '^', '*', '@', '.']:
            char = ' '

        # Add the character to the new string
        temp += char

    # Return the new string
    return temp


# This function handles the '' elements in a given list
def handleEmpty(target):
    # Create a new empty list to return back later
    temp = []

    # For each element in the target list
    for i in target:
        # If it's NOT '' then add it to the new list
        if i != '':
            temp.append(i)

    # Return the new list
    return temp


# This function handles stop words in a given list
def handleStopWords(target):
    # Create a list that contains all the stop words
    stopWords = stopwords.words('english')

    # Create a new empty list to return back later
    temp = []

    # For each element in the target lsit
    for i in target:
        # If it's not a stop word then add it to the new list
        if i not in stopWords:
            temp.append(i)

    # Return the new list
    return temp


# This function receives a list of words and return a new list that contains
# the original words and their synonyms.
# The limit number of synonyms is also specified.
def synonym(list, number):

    # Create a new empty list to return back later
    synonyms = []

    # For each word in the given list
    for i in range(len(list)):

        # Start counting the number of synonyms
        limit = 0

        # For each set in the set of sysnonyms
        for syn in wordnet.synsets(list[i]):

            # For each element in that set
            for l in syn.lemmas():

                # Add it to the new list
                synonyms.append(l.name())

                # If count reaches the limit then exit the loop
                if limit == number:
                    break

                # Increase the counting by 1
                limit += 1

    # Return the new list
    return synonyms


# This function recieves a string and create bigrams tupples
def collocations(text):
    # Set the standard point to rank the biagrams based on the theory of
    # "In English some words often come with each other"
    bigram_measures = nltk.collocations.BigramAssocMeasures()

    # Split the string into a list
    length = len(text.split())

    # Create the biagrams from the list
    finder = BigramCollocationFinder.from_words(nltk.word_tokenize(text))

    # Use the standard point define above to sort the list of biagrams
    # The set of words that often comes with each other will have the highest point
    # The list is sorted by descending order
    collocations_result = finder.nbest(bigram_measures.pmi, length)

    # Return the list of biagrams
    return collocations_result


# This function gives user recommendations based on the search history
def recommend(query, current_searches, top):

    # Create a new empty string to store all search history
    search_history_keywords = ''

    # Split the search string into a list
    split_query = query.split(' ')

    # For each Search in the list of all Searches that were made
    for search in current_searches:
        # Get the search string
        search_string = ' '.join(search.keywords)

        # Add that string into the search history with a space at the end
        search_history_keywords += search_string
        search_history_keywords += ' '

    # Analyze the search history to detect the pairs of words that often come with each other
    query_collocations = collocations(search_history_keywords)

    # Filter that collocation list to only get the ones with the highest point
    query_collocations = query_collocations[0:top]

    # For each pair in the list of collocation
    for collocation in query_collocations:
        # If it contains a word in the search string then provide it as a recommendation
        for word in split_query:
            if word in collocation:
                print("You might be searching for: ", ' '.join(list(collocation)))

# a jap bike and call myself axis motors tuba irwin
# This is the similarity function which takes the number of synonyms per word and
# the number of top results to return
def similarity(synNum, filterNum):

    # Create a list to store the synonyms
    synList = synonym(searchList, synNum)
    # Join that list to get it in string
    synString = ' '.join(synList)

    # Fetch the dataset
    newsgroups_train = fetch_20newsgroups(subset='train')

    # This dictionary stores the index of each line in the datset and
    # its similarity to the search string
    dictOri = {}

    # This dictionary stores the index of each line in the datset and
    # its similarity to the string of synonyms
    dictSyn = {}


    # For each line in the dataset
    for i in range(len(newsgroups_train.data)):
        # Get the line which is a string
        line = newsgroups_train.data[i]

        # Handle special characters in the line
        line = handleSpecialChar(line)

        # Convert the line to lower case and split it to a list
        array = line.lower().split(' ')
        array = list(array)

        # Handle '' elements in that list
        array = handleEmpty(array)

        # Handle stop words in that list
        array = handleStopWords(array)

        # Join the list to create the search string
        line = ' '.join(array)

        # This is the combination list of the search string and the line
        combine = [searchString, line]

        # Create the tool to vectorize the strings
        # This tool uses TF-IDF method
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

        # Vectorize the combination list of the search string and the line
        vectors = vectorizer.fit_transform(combine)

        # Calculate the similarity point of the 2 vectors above
        # The similarity point is calculated using similarity point
        array_cos = cosine_similarity(vectors[0], vectors[1])[0][0]

        # Add the result to the original dictionary
        dictOri[i] = array_cos

        # This is the combination list of the synonym string and the line
        combineSyn = [synString, line]

        # Vectorize the combination list of the synonym string and the line
        vectors_syn = vectorizer.fit_transform(combineSyn)

        # Calculate the similarity point of the 2 vectors above
        # The similarity point is calculated using similarity point
        array_cos_syn = cosine_similarity(vectors_syn[0], vectors_syn[1])[0][0]

        # Add the result to the synonym dictionary
        dictSyn[i] = array_cos_syn

    # Check the original dictionary
    for key in list(dictOri):
        # If similarity = 0 then remove the element
        if dictOri[key] == 0:
            del dictOri[key]

    # Check the synonym dictionary
    for key in list(dictSyn):
        # If similarity = 0 then remove the element
        if dictSyn[key] == 0:
            del dictSyn[key]

    # This is the final dictionary storing the result to be returned
    resultDict = {}

    # Make the result dictionary the intersection of
    # the original dictionay and the synonym dictionary
    for item in dictOri.keys():
        if item in dictSyn.keys():
            resultDict[item] = dictOri[item]

    # Filter the result dictionary and get the top ones
    top = Counter(resultDict).most_common(filterNum)
    print(top)


# This variable is used to continue/exit the while loop below
running = True

# This list contains all Searches
current_searches = []

# This while loop ends when user set running to false
while (running):

    # Ask user for the search string
    searchString = input("Give me your string:\n")

    # Split the search string into a list
    searchList = searchString.split(' ')

    # Create a new Search object
    new_search = Search(searchList)

    # Start measuring the time
    start = timer()

    # Set the number of synonym per word
    numSyn = 3

    # Set the filter number of top result
    numFilter = 10

    # Execute the similarity function
    similarity(numSyn, numFilter)

    # Stop measuring the time
    end = timer()

    # Get the execution time and print it out
    time = end - start
    print(time, " seconds")

    # Recommend user the string that they might want to seach
    recommend(searchString, current_searches, 4)

    # Add the search to the history
    current_searches.append(new_search)

    # Ask user whether he/she want to search again
    search_again = input("Search again? (Type y or yes to search again, otherwise type anything) \n")

    # If not then exit the loop
    if not search_again.lower().startswith('y'):
        break

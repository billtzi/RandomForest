import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import nltk
from sklearn.feature_extraction import stop_words

nltk.download('punkt')
sum_score = 0
#read from user how many times to run the training set
N = input("How many times do you want to run the test?")

my_stop_words = ['.', ',', '!', '_', '!', '?', '-', '\'', '\"', ';', '/', '\\', ']', '[', '{', '}']
#load files from folder data which is in the project
documents1 = load_files(r'data')
#delete stop words
#I think that it doesn't run correctly
documents = [word for word in documents1.data if word not in stop_words.ENGLISH_STOP_WORDS]
#delete my stop words
documents = [word for word in documents1.data if word not in my_stop_words]
#create vector
movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
movie_counts = movie_vec.fit_transform(documents)
tfidf_transformer = TfidfTransformer()
movie_tfidf = tfidf_transformer.fit_transform(movie_counts)
#for loop for training test
for i in range(int(N)):
    #select 80% for training and 20% for test
    docs_train, docs_test, y_train, y_test = train_test_split(movie_tfidf, documents1.target, test_size = 0.20, random_state = 12)
    #run random forest
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(docs_train, y_train)
    #take the prediction for the test
    y_pred = clf.predict(docs_test)
    #calculate the score
    score = sklearn.metrics.accuracy_score(y_test, y_pred)
    #sumarize the score to find the average
    sum_score = sum_score+score
    #print the score
    print((i+1), "Accuracy Score :", (score * 100.00), "%")

#calculate the average score
sum_score = sum_score / float(N) * 100.00
#print the averacy score
print("Average Accouracy Score For " + N + " times", sum_score, "%")
#read from user the number of his reviews

str = 'the users reviews'
while str != '':
    string = 'Give Review or press enter to continue'
    str = input(string)
    # add at the end of documents list the user's review
    if str != '':
        documents.append(str)
#delete the stop words
documents = [word for word in documents if word not in stop_words.ENGLISH_STOP_WORDS]
#delete my stop words
documents = [word for word in documents if word not in my_stop_words]

#create vector
movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
movie_counts = movie_vec.fit_transform(documents)
tfidf_transformer = TfidfTransformer()
reviews_tfidf = tfidf_transformer.fit_transform(movie_counts)
#take predictions for this reviews
pred = clf.predict(reviews_tfidf)
#print the predictions for user's reviews
i = 0
for review, category in zip(documents, pred):
    if i >= 2000:
        print('%r : %s' % (review, documents1.target_names[category]))
    i = i + 1

#this movie is too bad. i want to sleep
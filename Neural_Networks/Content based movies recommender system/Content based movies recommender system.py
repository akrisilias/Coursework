#!/usr/bin/env python
# coding: utf-8

# # Στοιχεία Ομάδας
# ## Ομάδα Α9
# <br>
# Μαύρος Γεώργιος<br>03112618<br><br>
# Κρίσιλιας Ανδρέας<br>03114778 

# # Εργαστηριακή Άσκηση 2. Μη επιβλεπόμενη μάθηση. 
# Ημερομηνία εκφώνησης άσκησης: 3/12/18
# ## Σύστημα συστάσεων βασισμένο στο περιεχόμενο
# ## Σημασιολογική απεικόνιση δεδομένων με χρήση SOM 
# 
# 

# In[ ]:


get_ipython().system(u'pip install --upgrade pip')
get_ipython().system(u'pip install --upgrade numpy')
get_ipython().system(u'pip install --upgrade pandas')
get_ipython().system(u'pip install --upgrade nltk')
get_ipython().system(u'pip install --upgrade scikit-learn')


# ## Εισαγωγή του Dataset

# Το σύνολο δεδομένων με το οποίο θα δουλέψουμε είναι βασισμένο στο [Carnegie Mellon Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/). Πρόκειται για ένα dataset με περίπου 40.000 περιγραφές ταινιών. Η περιγραφή κάθε ταινίας αποτελείται από τον τίτλο της, μια ή περισσότερες ετικέτες που χαρακτηρίζουν το είδος της ταινίας και τέλος τη σύνοψη της υπόθεσής της. Αρχικά εισάγουμε το dataset (χρησιμοποιήστε αυτούσιο τον κώδικα, δεν χρειάζεστε το αρχείο csv) στο dataframe `df_data_1`: 

# In[ ]:


import pandas as pd

dataset_url = "https://drive.google.com/uc?export=download&id=1PdkVDENX12tQliCk_HtUnAUbfxXvnWuG"
# make direct link for drive docs this way https://www.labnol.org/internet/direct-links-for-google-drive/28356/
df_data_1 = pd.read_csv(dataset_url, sep='\t',  header=None, quoting=3, error_bad_lines=False)


# Κάθε ομάδα θα δουλέψει σε ένα μοναδικό υποσύνολο 5.000 ταινιών (διαφορετικό dataset για κάθε ομάδα) ως εξής
# 
# 1. Κάθε ομάδα μπορεί να βρει [εδώ](https://docs.google.com/spreadsheets/d/12AmxMqvjrc0ruNmZYTBNxvnEktbec1DRG64LW7SX4HA/edit?usp=sharing) τον μοναδικό  αριθμό της "Seed" από 1 έως 128. 
# 
# 2. Το data frame `df_data_2` έχει 128 γραμμές (ομάδες) και 5.000 στήλες. Σε κάθε ομάδα αντιστοιχεί η γραμμή του πίνακα με το `team_seed_number` της. Η γραμμή αυτή θα περιλαμβάνει 5.000 διαφορετικούς αριθμούς που αντιστοιχούν σε ταινίες του αρχικού dataset. 
# 
# 3. Στο επόμενο κελί αλλάξτε τη μεταβλητή `team_seed_number` με το Seed της ομάδας σας από το Google Sheet.
# 
# 4. Τρέξτε τον κώδικα. Θα προκύψουν τα μοναδικά για κάθε ομάδα  titles, categories, catbins, summaries και corpus με τα οποία θα δουλέψετε.

# In[ ]:


import numpy as np

# team seed
team_seed_number = 9

movie_seeds_url = "https://drive.google.com/uc?export=download&id=1NkzL6rqv4DYxGY-XTKkmPqEoJ8fNbMk_"
df_data_2 = pd.read_csv(movie_seeds_url, header=None, error_bad_lines=False)

# sample of dataset to work with 
my_index = df_data_2.iloc[team_seed_number,:].values

titles = df_data_1.iloc[:, [2]].values[my_index] # movie titles (string)
categories = df_data_1.iloc[:, [3]].values[my_index] # movie categories (string)
bins = df_data_1.iloc[:, [4]]
catbins = bins[4].str.split(',', expand=True).values.astype(np.float)[my_index] # movie categories in binary form (1 feature per category)
summaries =  df_data_1.iloc[:, [5]].values[my_index] # movie summaries (string)
corpus = summaries[:,0].tolist() # list form of summaries


# - Ο πίνακας **titles** περιέχει τους τίτλους των ταινιών. Παράδειγμα: 'Sid and Nancy'.
# - O πίνακας **categories** περιέχει τις κατηγορίες (είδη) της ταινίας υπό τη μορφή string. Παράδειγμα: '"Tragedy",  "Indie",  "Punk rock",  "Addiction Drama",  "Cult",  "Musical",  "Drama",  "Biopic \[feature\]",  "Romantic drama",  "Romance Film",  "Biographical film"'. Παρατηρούμε ότι είναι μια comma separated λίστα strings, με κάθε string να είναι μια κατηγορία.
# - Ο πίνακας **catbins** περιλαμβάνει πάλι τις κατηγορίες των ταινιών αλλά σε δυαδική μορφή ([one hot encoding](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f)). Έχει διαστάσεις 5.000 x 322 (όσες οι διαφορετικές κατηγορίες). Αν η ταινία ανήκει στο συγκεκριμένο είδος η αντίστοιχη στήλη παίρνει την τιμή 1, αλλιώς παίρνει την τιμή 0.
# - Ο πίνακας **summaries** και η λίστα **corpus** περιλαμβάνουν τις συνόψεις των ταινιών (η corpus είναι απλά ο summaries σε μορφή λίστας). Κάθε σύνοψη είναι ένα (συνήθως μεγάλο) string. Παράδειγμα: *'The film is based on the real story of a Soviet Internal Troops soldier who killed his entire unit  as a result of Dedovschina. The plot unfolds mostly on board of the prisoner transport rail car guarded by a unit of paramilitary conscripts.'*
# - Θεωρούμε ως **ID** της κάθε ταινίας τον αριθμό γραμμής της ή το αντίστοιχο στοιχείο της λίστας. Παράδειγμα: για να τυπώσουμε τη σύνοψη της ταινίας με `ID=99` (την εκατοστή) θα γράψουμε `print(corpus[99])`.

# In[ ]:


ID = 99
print(titles[ID])
print(categories[ID])
print(catbins[ID])
print(corpus[ID])


# # Εφαρμογή 1. Υλοποίηση συστήματος συστάσεων ταινιών βασισμένο στο περιεχόμενο
# <img src="http://clture.org/wp-content/uploads/2015/12/Netflix-Streaming-End-of-Year-Posts.jpg" width="50%">

# Η πρώτη εφαρμογή που θα αναπτύξετε θα είναι ένα [σύστημα συστάσεων](https://en.wikipedia.org/wiki/Recommender_system) ταινιών βασισμένο στο περιεχόμενο (content based recommender system). Τα συστήματα συστάσεων στοχεύουν στο να προτείνουν αυτόματα στο χρήστη αντικείμενα από μια συλλογή τα οποία ιδανικά θέλουμε να βρει ενδιαφέροντα ο χρήστης. Η κατηγοριοποίηση των συστημάτων συστάσεων βασίζεται στο πώς γίνεται η επιλογή (filtering) των συστηνόμενων αντικειμένων. Οι δύο κύριες κατηγορίες είναι η συνεργατική διήθηση (collaborative filtering) όπου το σύστημα προτείνει στο χρήστη αντικείμενα που έχουν αξιολογηθεί θετικά από χρήστες που έχουν παρόμοιο με αυτόν ιστορικό αξιολογήσεων και η διήθηση με βάση το περιεχόμενο (content based filtering), όπου προτείνονται στο χρήστη αντικείμενα με παρόμοιο περιεχόμενο (με βάση κάποια χαρακτηριστικά) με αυτά που έχει προηγουμένως αξιολογήσει θετικά.
# 
# Το σύστημα συστάσεων που θα αναπτύξετε θα βασίζεται στο **περιεχόμενο** και συγκεκριμένα στις συνόψεις των ταινιών (corpus). 
# 

# ## Μετατροπή σε TFIDF
# 
# Το πρώτο βήμα θα είναι λοιπόν να μετατρέψετε το corpus σε αναπαράσταση tf-idf:

# In[ ]:


# transform corpus to tf-idf representation
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
corpus_tf_idf = vectorizer.transform(corpus)


# Η συνάρτηση [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) όπως καλείται εδώ **δεν είναι βελτιστοποιημένη**. Οι επιλογές των μεθόδων και παραμέτρων της μπορεί να έχουν **δραματική επίδραση στην ποιότητα των συστάσεων** και είναι διαφορετικές για κάθε dataset. Επίσης, οι επιλογές αυτές έχουν πολύ μεγάλη επίδραση και στη **διαστατικότητα και όγκο των δεδομένων**. Η διαστατικότητα των δεδομένων με τη σειρά της θα έχει πολύ μεγάλη επίδραση στους **χρόνους εκπαίδευσης**, ιδιαίτερα στη δεύτερη εφαρμογή της άσκησης. Ανατρέξτε στα notebooks του εργαστηρίου και στο [FAQ](https://docs.google.com/document/d/1jL4gRag_LHbVCYIt5XVJ53iJPb6RZWi02rT5mPXiqEU/edit?usp=sharing) των ασκήσεων.
# 

# In[ ]:


print(corpus_tf_idf.shape)


# ## Υλοποίηση του συστήματος συστάσεων
# 
# Το σύστημα συστάσεων που θα παραδώσετε θα είναι μια συνάρτηση `content_recommender` με δύο ορίσματα `target_movie` και `max_recommendations`. Στην `target_movie` περνάμε το ID μιας ταινίας-στόχου για την οποία μας ενδιαφέρει να βρούμε παρόμοιες ως προς το περιεχόμενο (τη σύνοψη) ταινίες, `max_recommendations` στο πλήθος.
# Υλοποιήστε τη συνάρτηση ως εξής: 
# - για την ταινία-στόχο, από το `corpus_tf_idf` υπολογίστε την [ομοιότητα συνημιτόνου](https://en.wikipedia.org/wiki/Cosine_similarity) της με όλες τις ταινίες της συλλογής σας
# - με βάση την ομοιότητα συνημιτόνου που υπολογίσατε, δημιουργήστε ταξινομημένο πίνακα από το μεγαλύτερο στο μικρότερο, με τα indices (`ID`) των ταινιών. Παράδειγμα: αν η ταινία με index 1 έχει ομοιότητα συνημιτόνου με 3 ταινίες \[0.2 1 0.6\] (έχει ομοιότητα 1 με τον εαύτό της) ο ταξινομημένος αυτός πίνακας indices θα είναι \[1 2 0\].
# - Για την ταινία-στόχο εκτυπώστε: id, τίτλο, σύνοψη, κατηγορίες (categories)
# - Για τις `max_recommendations` ταινίες (πλην της ίδιας της ταινίας-στόχου που έχει cosine similarity 1 με τον εαυτό της) με τη μεγαλύτερη ομοιότητα συνημιτόνου (σε φθίνουσα σειρά), τυπώστε σειρά σύστασης (1 πιο κοντινή, 2 η δεύτερη πιο κοντινή κλπ), id, τίτλο, σύνοψη, κατηγορίες (categories)
# 

# In[ ]:


# library for cosine similarity
import scipy as sp

def content_recommender(target_movie, max_recommendations):
    # number of movies
    n_movies = corpus_tf_idf.shape[0]
    # cosine similarity calculation
    cossim = np.zeros(n_movies)
    for i in range (n_movies):
        cossim[i] = 1 - sp.spatial.distance.cosine(corpus_tf_idf[target_movie].todense(), corpus_tf_idf[i].todense())
    # descending ordered indices
    max_recommendations = min(max_recommendations, n_movies-1)
    indsim = (-cossim).argsort()[:max_recommendations+1]
    # print id, title, summary, categories for target movie
    print("The target movie")
    print("Has ID:", target_movie)
    print("Has title:", titles[target_movie][0])
    print("Belongs to the categories:", categories[target_movie])
    print("And has the following summary:", summaries[target_movie][0])
    # print id, title, summary, categories for recommended movies
    for i in range (1, max_recommendations+1):
        ind = indsim[i]
        print("\n")
        print("Reccomendation Νο.", i)
        print("ID:", ind)
        print("Title:", titles[ind][0])
        print("Categoires:", categories[ind])
        print("Summary:", summaries[ind][0])  
        
        


# ## Βελτιστοποίηση
# 
# Αφού υλοποιήσετε τη συνάρτηση `content_recommender` χρησιμοποιήστε τη για να βελτιστοποιήσετε την `TfidfVectorizer`. Συγκεκριμένα, αρχικά μπορείτε να δείτε τι επιστρέφει το σύστημα για τυχαίες ταινίες-στόχους και για ένα μικρό `max_recommendations` (2 ή 3). Αν σε κάποιες ταινίες το σύστημα μοιάζει να επιστρέφει σημασιολογικά κοντινές ταινίες σημειώστε το `ID` τους. Δοκιμάστε στη συνέχεια να βελτιστοποιήσετε την `TfidfVectorizer` για τα συγκεκριμένα `ID` ώστε να επιστρέφονται σημασιολογικά κοντινές ταινίες για μεγαλύτερο αριθμό `max_recommendations`. Παράλληλα, όσο βελτιστοποιείτε την `TfidfVectorizer`, θα πρέπει να λαμβάνετε καλές συστάσεις για μεγαλύτερο αριθμό τυχαίων ταινιών. Μπορείτε επίσης να βελτιστοποιήσετε τη συνάρτηση παρατηρώντας πολλά φαινόμενα που το σύστημα εκλαμβάνει ως ομοιότητα περιεχομένου ενώ επί της ουσίας δεν είναι επιθυμητό να συνυπολογίζονται (δείτε σχετικά το [FAQ](https://docs.google.com/document/d/1jL4gRag_LHbVCYIt5XVJ53iJPb6RZWi02rT5mPXiqEU/edit?usp=sharing)). Ταυτόχρονα, μια άλλη κατεύθυνση της βελτιστοποίησης είναι να χρησιμοποιείτε τις παραμέτρους του `TfidfVectorizer` έτσι ώστε να μειώνονται οι διαστάσεις του Vector Space Model μέχρι το σημείο που θα αρχίσει να εμφανίζονται επιπτώσεις στην ποιότητα των συστάσεων. 
# 
# 
# 

# In[ ]:


import nltk

from nltk.corpus import stopwords
import string

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

import collections

# needed for tokenizer
nltk.download('punkt')

# download file with English stopwords
nltk.download('stopwords') 

# needed downloads for stemmer/lemmatizer
nltk.download('wordnet') 
nltk.download('rslp')

# words to be ignored
extra_words = ['plot', 'film', 'story', 'based']
found_names = ['David', 'Scott', 'Alice', 'Veera', 'Ashok', 'Frank', 'Anna', 'Raj', 'Ravi', 'Rahul', 'Ben', 'Jack', 'Max', 'Jake', 'Nick', 'John', 'Barney', 'George']
common_names = ['Sam', 'Jack', 'Paul', 'Sarah', 'Charles', 'John', 'Mary', 'Mike', 'Michael', 'Alex', 'Joe', 'Peter', 'Mark', 'Jimmy', 'Nick', 'Lucy', 'Rachel', 'Tony', 'Claire', 'Eddie', 'Emily', 'Lisa', 'Bobby', 'William', 'Maria', 'Kate', 'Billy', 'Ray', 'Adam', 'Tommy', 'Danny', 'Steve', 'Bob', 'Henry', 'Jake', 'Laura', 'Dave', 'Jenny', 'Amy', 'Maggie', 'Kevin', 'Simon', 'Bill', 'Karen', 'Sophie', 'Annie', 'Eric', 'Matt', 'Jane', 'Chris', 'Marie', 'Jason', 'Larry', 'Carl', 'Linda', 'Richard', 'Josh', 'Grace', 'Robert', 'Kelly', 'Daniel', 'Pete', 'Harry', 'Martin', 'Phil', 'Rose', 'Helen', 'Tim']
useless_words = extra_words + stopwords.words('english') + list(string.punctuation) + found_names + common_names
useless_words = [word.lower() for word in useless_words]

# remove words with lenght greater than 1
def thorough_filter(words):
    filtered_words = []
    for word in words:
        pun = []
        for letter in word:
            pun.append(letter in string.punctuation)
        if not all(pun):
            filtered_words.append(word)
    return filtered_words


# In[ ]:


def opt_TfidfVectorizer(docs):
    # convert to lower case letters
    for i in range(len(docs)):
        docs[i] = docs[i].lower()
    # tokenize    
    wordss = [nltk.word_tokenize(doc) for doc in docs]
    for i in range(len(wordss)):
        # ignore words
        wordss[i] = [word for word in wordss[i] if word not in useless_words]
        wordss[i] = thorough_filter(wordss[i])
        # stemming
        wordss[i] = [porter_stemmer.stem(word) for word in wordss[i]]
        # untokenize to pass from TfidfVectorizer
        wordss[i] = ' '.join(wordss[i])
    # min_df=k --> ignore terms with document frequency k-1 and below   
    opt_vectorizer = TfidfVectorizer(min_df=3)
    opt_vectorizer.fit(wordss)
    tf_idf = opt_vectorizer.transform(wordss)
    return tf_idf
    


# In[ ]:


corpus_tf_idf = opt_TfidfVectorizer(corpus)


# In[ ]:


print(corpus_tf_idf.shape)


# ## Επεξήγηση επιλογών και ποιοτική ερμηνεία
# 
# Σε markdown περιγράψτε πώς προχωρήσατε στις επιλογές σας για τη βελτιστοποίηση της `TfidfVectorizer`. Επίσης σε markdown δώστε 10 παραδείγματα (IDs) από τη συλλογή σας που επιστρέφουν καλά αποτελέσματα μέχρι `max_recommendations` (5 και παραπάνω) και σημειώστε συνοπτικά ποια είναι η θεματική που ενώνει τις ταινίες.
# 
# Δείτε [εδώ](https://pastebin.com/raw/ZEvg5t3z) ένα παράδειγμα εξόδου του βελτιστοποιημένου συστήματος συστάσεων για την ταίνία ["Q Planes"](https://en.wikipedia.org/wiki/Q_Planes) με την κλήση της συνάρτησης για κάποιο seed `content_recommender(529,3)`. Είναι φανερό ότι η κοινή θεματική των ταινιών είναι τα αεροπλάνα, οι πτήσεις, οι πιλότοι, ο πόλεμος.

# ### *Περιγραφή επιλογών*
# Αρχικά μετατρέψαμε όλους του κεφαλαίους χαρακτήρες σε πεζούς, ώστε να μη λογίζονται ως διαφορετικές λέξεις κάποιες που τη μία ξεκινάνε με κεφαλαίο και την άλλη με μικρό (π.χ. Nowadays, nowadays). Αυτό βέβαια το έκανε και από μόνος του ο TfidfVectorizer, επομένως δε λαμβάνεται ως βελτιστοποίηση.<br>
# Εν συνεχεία, ορίσαμε κάποιες stop words για να αγνοηθούν αφού δεν προσδίδουν χρήσιμη πληροφορία και απλά μεγαλώνουν τις διαστάσεις του tf_idf διανύσματος. Τέτοιες λέξεις είναι διάφοροι συχνά χρησιμοποιούμενοι σύνδεσμοι, αντωνυμίες, επιρρήματα, σημεία στίξης και λέξεις που εμφανίζονται συχνά λόγω της θεματολογίας (film, plot, story, etc). Αφαιρέσαμε επίσης αρκετά μικρά ονόματα (David, Alice, Max, κλπ) καθότι όταν εμφανίζονται συχνά σε μία σύνοψη, πολλές φορές γίνεται σύσταση μία ταινία κυρίως βάση αυτών, χωρίς ωστόσο να υπάρχει κάποια ομοιότητα στην πλοκή με την ταινία στόχο. Κάποια first names βέβαια δεν τα βάλαμε εσκεμμένα στα stop words (e.g. Tom & Jerry) διότι σε κάποιες περιπτώσεις βοηθάνε στην εύρεση καλών συστάσεων.<br>
# Ύστερα εφαρμόσαμε stemming, ώστε να κρατήσουμε μόνο τις ρίζες των λέξεων, ώστε όροι εννοιολογικά κοντά να λογίζονται ως ίδιοι και να μικρύνουμε έτσι ακόμα περισσότερο τις διαστάσεις.<br>
# Τέλος για να μικρύνουμε ακόμα περισσότερο τις διαστάσεις, κρατήσαμε μόνο όρους που εμφανίζονται σε τουλάχιστον 3 συνόψεις. Η απώλεια όρων με document frequency 2 θεωρήθηκε αρκετά ασήμαντη μπροστά στο πλήθος των ταινιών (5000), και η απώλεια όρων με document frequency 1 ήταν ούτως ή άλλως απαραίτητη αφού δεν προσέδιδε κάποια χρήσιμη πληροφορία και αύξανε τις διαστάσεις χωρίς λόγο. 

# ### *Παραδείγματα*
# 1. 832 : Elmer and Bugs<br>
# 2. 817 : Murders and sheriff<br>
# 3. 3882 : Tom and Jerry<br>
# 4. 217 : Tora-San<br>
# 5. 3439 : Daffy Duck<br>
# 6. 4129 : Soviet Union<br>
# 7. 2047 : Mickey and Minnie<br>
# 8. 4297 : Santa and Christmas<br>
# 9. 14 : Charlie Brown<br>
# 10. 2702 : Bugs Bunny

# ## Tip: persistence αντικειμένων με joblib.dump
# 
# H βιβλιοθήκη [joblib](https://pypi.python.org/pypi/joblib) της Python δίνει κάποιες εξαιρετικά χρήσιμες ιδιότητες στην ανάπτυξη κώδικα: pipelining, παραλληλισμό, caching και variable persistence. Τις τρεις πρώτες ιδιότητες τις είδαμε στην πρώτη άσκηση. Στην παρούσα άσκηση θα μας φανεί χρήσιμη η τέταρτη, το persistence των αντικειμένων. Συγκεκριμένα μπορούμε με:
# 
# ```python
# from sklearn.externals import joblib  
# joblib.dump(my_object, 'my_object.pkl') 
# ```
# 
# να αποθηκεύσουμε οποιοδήποτε αντικείμενο-μεταβλητή (εδώ το `my_object`) απευθείας πάνω στο filesystem ως αρχείο, το οποίο στη συνέχεια μπορούμε να ανακαλέσουμε ως εξής:
# 
# ```python
# my_object = joblib.load('my_object.pkl')
# ```
# 
# Μπορούμε έτσι να ανακαλέσουμε μεταβλητές ακόμα και αφού κλείσουμε και ξανανοίξουμε το notebook, χωρίς να χρειαστεί να ακολουθήσουμε ξανά όλα τα βήματα ένα - ένα για την παραγωγή τους, κάτι ιδιαίτερα χρήσιμο αν αυτή η διαδικασία είναι χρονοβόρα. Προσοχή: αυτό ισχύει μόνο στα Azure και Kaggle, στο Colab και στο IBM τα αρχεία εξαφανίζονται όταν ανακυκλώνεται ο πυρήνας και θα πρέπει να τα αποθηκεύετε τοπικά. Περισσότερα στο [FAQ](https://docs.google.com/document/d/1jL4gRag_LHbVCYIt5XVJ53iJPb6RZWi02rT5mPXiqEU/edit?usp=sharing).
# 
# Ας αποθηκεύσουμε το `corpus_tf_idf` και στη συνέχεια ας το ανακαλέσουμε.

# In[ ]:


from sklearn.externals import joblib
joblib.dump(corpus_tf_idf, 'corpus_tf_idf.pkl') 


# 
# 
# Μπορείτε με ένα απλό `!ls` να δείτε ότι το αρχείο `corpus_tf_idf.pkl` υπάρχει στο filesystem σας (== persistence):

# In[ ]:


get_ipython().system(u'ls -lh')


# και μπορούμε να τα διαβάσουμε με `joblib.load`

# In[ ]:


from sklearn.externals import joblib
corpus_tf_idf = joblib.load('corpus_tf_idf.pkl')


# # Εφαρμογή 2.  Σημασιολογική απεικόνιση της συλλογής ταινιών με χρήση SOM
# <img src="http://visual-memory.co.uk/daniel/Documents/intgenre/Images/film-genres.jpg" width="35%">

# ## Δημιουργία dataset
# Στη δεύτερη εφαρμογή θα βασιστούμε στις τοπολογικές ιδιότητες των Self Organizing Maps (SOM) για να φτιάξουμε ενά χάρτη (grid) δύο διαστάσεων όπου θα απεικονίζονται όλες οι ταινίες της συλλογής της ομάδας με τρόπο χωρικά συνεκτικό ως προς το περιεχόμενο και κυρίως το είδος τους. 
# 
# Η `build_final_set` αρχικά μετατρέπει την αραιή αναπαράσταση tf-idf της εξόδου της `TfidfVectorizer()` σε πυκνή (η [αραιή αναπαράσταση](https://en.wikipedia.org/wiki/Sparse_matrix) έχει τιμές μόνο για τα μη μηδενικά στοιχεία). 
# 
# Στη συνέχεια ενώνει την πυκνή `dense_tf_idf` αναπαράσταση και τις binarized κατηγορίες `catbins` των ταινιών ως επιπλέον στήλες (χαρακτηριστικά). Συνεπώς, κάθε ταινία αναπαρίσταται στο Vector Space Model από τα χαρακτηριστικά του TFIDF και τις κατηγορίες της.
# 
# Τέλος, δέχεται ένα ορισμα για το πόσες ταινίες να επιστρέψει, με default τιμή όλες τις ταινίες (5000). Αυτό είναι χρήσιμο για να μπορείτε αν θέλετε να φτιάχνετε μικρότερα σύνολα δεδομένων ώστε να εκπαιδεύεται ταχύτερα το SOM.
# 
# Σημειώστε ότι το IBM Watson δείνει "Kernel dead" εάν δεν έχετε βελτιστοποιήσει το tfidf και μικρύνει τις διαστάσεις του dataset (πιθανότατα κάποια υπέρβαση μνήμης).

# In[ ]:


def build_final_set(doc_limit = 5000, tf_idf_only=False):
    # convert sparse tf_idf to dense tf_idf representation
    dense_tf_idf = corpus_tf_idf.toarray()[0:doc_limit,:]
    if tf_idf_only:
        # use only tf_idf
        final_set = dense_tf_idf
    else:
        # append the binary categories features horizontaly to the (dense) tf_idf features
        final_set = np.hstack((dense_tf_idf, catbins[0:doc_limit,:]))
        # somoclu needs float32 data
    return np.array(final_set, dtype=np.float32)


# In[ ]:


final_set = build_final_set()


# Τυπώνουμε τις διαστάσεις του τελικού dataset μας. Χωρίς βελτιστοποίηση του TFIDF θα έχουμε περίπου 50.000 χαρακτηριστικά.

# In[ ]:


final_set.shape


# Με βάση την εμπειρία σας στην προετοιμασία των δεδομένων στην επιβλεπόμενη μάθηση, υπάρχει κάποιο βήμα προεπεξεργασίας που θα μπορούσε να εφαρμοστεί σε αυτό το dataset; 

# *Κάποιο βήμα προεπεξεργασίας που θα μπορούσε να εφαρμοστεί είναι η επιλογή χαρακτηριστικών, έτσι ώστε να ξεφορτωθούμε χαρακτηριστικά που έχουν μικρή διακύμανση και δε χρησιμεύουν ιδιαίτερα στην ταξινόμηση.* 

# ## Εκπαίδευση χάρτη SOM
# 
# Θα δουλέψουμε με τη βιβλιοθήκη SOM ["Somoclu"](http://somoclu.readthedocs.io/en/stable/index.html). Εισάγουμε τις somoclu και matplotlib και λέμε στη matplotlib να τυπώνει εντός του notebook (κι όχι σε pop up window).

# In[ ]:


# install somoclu
get_ipython().system(u'pip install --upgrade somoclu')
# import sompoclu, matplotlib
import somoclu
import matplotlib
# we will plot inside the notebook and not in separate window
get_ipython().magic(u'matplotlib inline')


# Καταρχάς διαβάστε το [function reference](http://somoclu.readthedocs.io/en/stable/reference.html) του somoclu. Θα δoυλέψουμε με χάρτη τύπου planar, παραλληλόγραμμου σχήματος νευρώνων με τυχαία αρχικοποίηση (όλα αυτά είναι default). Μπορείτε να δοκιμάσετε διάφορα μεγέθη χάρτη ωστόσο όσο ο αριθμός των νευρώνων μεγαλώνει, μεγαλώνει και ο χρόνος εκπαίδευσης. Για το training δεν χρειάζεται να ξεπεράσετε τα 100 epochs. Σε γενικές γραμμές μπορούμε να βασιστούμε στις default παραμέτρους μέχρι να έχουμε τη δυνατότητα να οπτικοποιήσουμε και να αναλύσουμε ποιοτικά τα αποτελέσματα. Ξεκινήστε με ένα χάρτη 10 x 10, 100 epochs training και ένα υποσύνολο των ταινιών (π.χ. 2000). Χρησιμοποιήστε την `time` για να έχετε μια εικόνα των χρόνων εκπαίδευσης. Ενδεικτικά, με σωστή κωδικοποίηση tf-idf, μικροί χάρτες για λίγα δεδομένα (1000-2000) παίρνουν γύρω στο ένα λεπτό ενώ μεγαλύτεροι χάρτες με όλα τα δεδομένα μπορούν να πάρουν 10-15 λεπτά ή και περισσότερο.
# 

# In[ ]:


# initialize Somoclu with 30x30 map 
n_columns, n_rows = 30, 30
som = somoclu.Somoclu(n_columns, n_rows)

# import time to measure
import time

# train the final set
start_time = time.time()
som.train(final_set, epochs=100)
end_time = time.time()

# print training time
train_time = (end_time - start_time) / 60
print("Training time for the final set is :", train_time, "minutes")


# 
# ## Best matching units
# 
# Μετά από κάθε εκπαίδευση αποθηκεύστε σε μια μεταβλητή τα best matching units (bmus) για κάθε ταινία. Τα bmus μας δείχνουν σε ποιο νευρώνα ανήκει η κάθε ταινία. Προσοχή: η σύμβαση των συντεταγμένων των νευρώνων είναι (στήλη, γραμμή) δηλαδή το ανάποδο από την Python. Με χρήση της [np.unique](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.unique.html) (μια πολύ χρήσιμη συνάρτηση στην άσκηση) αποθηκεύστε τα μοναδικά best matching units και τους δείκτες τους (indices) προς τις ταινίες. Σημειώστε ότι μπορεί να έχετε λιγότερα μοναδικά bmus από αριθμό νευρώνων γιατί μπορεί σε κάποιους νευρώνες να μην έχουν ανατεθεί ταινίες. Ως αριθμό νευρώνα θα θεωρήσουμε τον αριθμό γραμμής στον πίνακα μοναδικών bmus.
# 

# In[ ]:


# calculate bmus
bmus = som.bmus

# print bmus
print("movies bmus ( dimensions:", bmus.shape,"):\n", bmus)

# save unique bmus and their indices
ubmus, indices = np.unique(bmus, return_inverse=True, axis=0)
indices = indices.tolist()
print("\Unique bmus ( dimensions:", ubmus.shape,"):\n", ubmus)
print("\nUnique bmus indices ( length:", len(indices),"):\n", indices)


# 
# ## Ομαδοποίηση (clustering)
# 
# Τυπικά, η ομαδοποίηση σε ένα χάρτη SOM προκύπτει από το unified distance matrix (U-matrix): για κάθε κόμβο υπολογίζεται η μέση απόστασή του από τους γειτονικούς κόμβους. Εάν χρησιμοποιηθεί μπλε χρώμα στις περιοχές του χάρτη όπου η τιμή αυτή είναι χαμηλή (μικρή απόσταση) και κόκκινο εκεί που η τιμή είναι υψηλή (μεγάλη απόσταση), τότε μπορούμε να πούμε ότι οι μπλε περιοχές αποτελούν clusters και οι κόκκινες αποτελούν σύνορα μεταξύ clusters.
# 
# To somoclu δίνει την επιπρόσθετη δυνατότητα να κάνουμε ομαδοποίηση των νευρώνων χρησιμοποιώντας οποιονδήποτε αλγόριθμο ομαδοποίησης του scikit-learn. Στην άσκηση θα χρησιμοποιήσουμε τον k-Means. Για τον αρχικό σας χάρτη δοκιμάστε ένα k=20 ή 25. Οι δύο προσεγγίσεις ομαδοποίησης είναι διαφορετικές, οπότε περιμένουμε τα αποτελέσματα να είναι κοντά αλλά όχι τα ίδια.
# 

# In[ ]:


# initialize KMeans
from sklearn.cluster import KMeans
# number of clusters
k = 30
kmeans = KMeans(n_clusters=k)

# clustering 
som.cluster(algorithm=kmeans)


# 
# ## Αποθήκευση του SOM
# 
# Επειδή η αρχικοποίηση του SOM γίνεται τυχαία και το clustering είναι και αυτό στοχαστική διαδικασία, οι θέσεις και οι ετικέτες των νευρώνων και των clusters θα είναι διαφορετικές κάθε φορά που τρέχετε τον χάρτη, ακόμα και με τις ίδιες παραμέτρους. Για να αποθηκεύσετε ένα συγκεκριμένο som και clustering χρησιμοποιήστε και πάλι την `joblib`. Μετά την ανάκληση ενός SOM θυμηθείτε να ακολουθήσετε τη διαδικασία για τα bmus.
# 

# In[ ]:


from sklearn.externals import joblib
joblib.dump(som, 'som_n30_k30_all_samples.pkl') 


# In[ ]:


from sklearn.externals import joblib
som = joblib.load('som_n30_k30_all_samples.pkl')


# 
# ## Οπτικοποίηση U-matrix, clustering και μέγεθος clusters
# 
# Για την εκτύπωση του U-matrix χρησιμοποιήστε τη `view_umatrix` με ορίσματα `bestmatches=True` και `figsize=(15, 15)` ή `figsize=(20, 20)`. Τα διαφορετικά χρώματα που εμφανίζονται στους κόμβους αντιπροσωπεύουν τα διαφορετικά clusters που προκύπτουν από τον k-Means. Μπορείτε να εμφανίσετε τη λεζάντα του U-matrix με το όρισμα `colorbar`. Μην τυπώνετε τις ετικέτες (labels) των δειγμάτων, είναι πολύ μεγάλος ο αριθμός τους.
# 
# Για μια δεύτερη πιο ξεκάθαρη οπτικοποίηση του clustering τυπώστε απευθείας τη μεταβλητή `clusters`.
# 
# Τέλος, χρησιμοποιώντας πάλι την `np.unique` (με διαφορετικό όρισμα) και την `np.argsort` (υπάρχουν και άλλοι τρόποι υλοποίησης) εκτυπώστε τις ετικέτες των clusters (αριθμοί από 0 έως k-1) και τον αριθμό των νευρώνων σε κάθε cluster, με φθίνουσα ή αύξουσα σειρά ως προς τον αριθμό των νευρώνων. Ουσιαστικά είναι ένα εργαλείο για να βρίσκετε εύκολα τα μεγάλα και μικρά clusters. 
# 
# Ακολουθεί ένα μη βελτιστοποιημένο παράδειγμα για τις τρεις προηγούμενες εξόδους:
# 
# <img src="https://image.ibb.co/i0tsfR/umatrix_s.jpg" width="35%">
# <img src="https://image.ibb.co/nLgHEm/clusters.png" width="35%">
# 
# 

# In[ ]:


import numpy as np

# umatrix
print("U-matrix:")
som.view_umatrix(bestmatches=True, colorbar=True, figsize=(15,15))

# clusters
print("\nClusters in numpy array format:")
print(som.clusters)

# sorted labels of clusters and number of neurons per cluster in descending order
lb_clusters, nr_neurons = np.unique(som.clusters, return_counts=True)
ind = np.argsort(-nr_neurons)
sorted_lb_clusters = [lb_clusters[i] for i in ind]
sorted_nr_neurons = [nr_neurons[i] for i in ind]
sorted_total = np.row_stack((sorted_lb_clusters, sorted_nr_neurons))
print("\nClusters sorted by decreasing number of neurons:")
print("Cluster index\nNumber of neurons\n", sorted_total)


# 
# ## Σημασιολογική ερμηνεία των clusters
# 
# Προκειμένου να μελετήσουμε τις τοπολογικές ιδιότητες του SOM και το αν έχουν ενσωματώσει σημασιολογική πληροφορία για τις ταινίες διαμέσου της διανυσματικής αναπαράστασης με το tf-idf και των κατηγοριών, χρειαζόμαστε ένα κριτήριο ποιοτικής επισκόπησης των clusters. Θα υλοποιήσουμε το εξής κριτήριο: Λαμβάνουμε όρισμα έναν αριθμό (ετικέτα) cluster. Για το cluster αυτό βρίσκουμε όλους τους νευρώνες που του έχουν ανατεθεί από τον k-Means. Για όλους τους νευρώνες αυτούς βρίσκουμε όλες τις ταινίες που τους έχουν ανατεθεί (για τις οποίες αποτελούν bmus). Για όλες αυτές τις ταινίες τυπώνουμε ταξινομημένη τη συνολική στατιστική όλων των ειδών (κατηγοριών) και τις συχνότητές τους. Αν το cluster διαθέτει καλή συνοχή και εξειδίκευση, θα πρέπει κάποιες κατηγορίες να έχουν σαφώς μεγαλύτερη συχνότητα από τις υπόλοιπες. Θα μπορούμε τότε να αναθέσουμε αυτήν/ές την/τις κατηγορία/ες ως ετικέτες κινηματογραφικού είδους στο cluster.
# 
# Μπορείτε να υλοποιήσετε τη συνάρτηση αυτή όπως θέλετε. Μια πιθανή διαδικασία θα μπορούσε να είναι η ακόλουθη:
# 
# 1. Ορίζουμε συνάρτηση `print_categories_stats` που δέχεται ως είσοδο λίστα με ids ταινιών. Δημιουργούμε μια κενή λίστα συνολικών κατηγοριών. Στη συνέχεια, για κάθε ταινία επεξεργαζόμαστε το string `categories` ως εξής: δημιουργούμε μια λίστα διαχωρίζοντας το string κατάλληλα με την `split` και αφαιρούμε τα whitespaces μεταξύ ετικετών με την `strip`. Προσθέτουμε τη λίστα αυτή στη συνολική λίστα κατηγοριών με την `extend`. Τέλος χρησιμοποιούμε πάλι την `np.unique` για να μετρήσουμε συχνότητα μοναδικών ετικετών κατηγοριών και ταξινομούμε με την `np.argsort`. Τυπώνουμε τις κατηγορίες και τις συχνότητες εμφάνισης ταξινομημένα. Χρήσιμες μπορεί να σας φανούν και οι `np.ravel`, `np.nditer`, `np.array2string` και `zip`.
# 
# 2. Ορίζουμε τη βασική μας συνάρτηση `print_cluster_neurons_movies_report` που δέχεται ως όρισμα τον αριθμό ενός cluster. Με τη χρήση της `np.where` μπορούμε να βρούμε τις συντεταγμένες των bmus που αντιστοιχούν στο cluster και με την `column_stack` να φτιάξουμε έναν πίνακα bmus για το cluster. Προσοχή στη σειρά (στήλη - σειρά) στον πίνακα bmus. Για κάθε bmu αυτού του πίνακα ελέγχουμε αν υπάρχει στον πίνακα μοναδικών bmus που έχουμε υπολογίσει στην αρχή συνολικά και αν ναι προσθέτουμε το αντίστοιχο index του νευρώνα σε μια λίστα. Χρήσιμες μπορεί να είναι και οι `np.rollaxis`, `np.append`, `np.asscalar`. Επίσης πιθανώς να πρέπει να υλοποιήσετε ένα κριτήριο ομοιότητας μεταξύ ενός bmu και ενός μοναδικού bmu από τον αρχικό πίνακα bmus.
# 
# 3. Υλοποιούμε μια βοηθητική συνάρτηση `neuron_movies_report`. Λαμβάνει ένα σύνολο νευρώνων από την `print_cluster_neurons_movies_report` και μέσω της `indices` φτιάχνει μια λίστα με το σύνολο ταινιών που ανήκουν σε αυτούς τους νευρώνες. Στο τέλος καλεί με αυτή τη λίστα την `print_categories_stats` που τυπώνει τις στατιστικές των κατηγοριών.
# 
# Μπορείτε βέβαια να προσθέσετε οποιαδήποτε επιπλέον έξοδο σας βοηθάει. Μια χρήσιμη έξοδος είναι πόσοι νευρώνες ανήκουν στο cluster και σε πόσους και ποιους από αυτούς έχουν ανατεθεί ταινίες.
# 
# Θα επιτελούμε τη σημασιολογική ερμηνεία του χάρτη καλώντας την `print_cluster_neurons_movies_report` με τον αριθμός ενός cluster που μας ενδιαφέρει. 
# 
# Παράδειγμα εξόδου για ένα cluster (μη βελτιστοποιημένος χάρτης, ωστόσο βλέπετε ότι οι μεγάλες κατηγορίες έχουν σημασιολογική  συνάφεια):
# 
# ```
# Overall Cluster Genres stats:  
# [('"Horror"', 86), ('"Science Fiction"', 24), ('"B-movie"', 16), ('"Monster movie"', 10), ('"Creature Film"', 10), ('"Indie"', 9), ('"Zombie Film"', 9), ('"Slasher"', 8), ('"World cinema"', 8), ('"Sci-Fi Horror"', 7), ('"Natural horror films"', 6), ('"Supernatural"', 6), ('"Thriller"', 6), ('"Cult"', 5), ('"Black-and-white"', 5), ('"Japanese Movies"', 4), ('"Short Film"', 3), ('"Drama"', 3), ('"Psychological thriller"', 3), ('"Crime Fiction"', 3), ('"Monster"', 3), ('"Comedy"', 2), ('"Western"', 2), ('"Horror Comedy"', 2), ('"Archaeology"', 2), ('"Alien Film"', 2), ('"Teen"', 2), ('"Mystery"', 2), ('"Adventure"', 2), ('"Comedy film"', 2), ('"Combat Films"', 1), ('"Chinese Movies"', 1), ('"Action/Adventure"', 1), ('"Gothic Film"', 1), ('"Costume drama"', 1), ('"Disaster"', 1), ('"Docudrama"', 1), ('"Film adaptation"', 1), ('"Film noir"', 1), ('"Parody"', 1), ('"Period piece"', 1), ('"Action"', 1)]```
#    

# In[ ]:


# 1.
def print_categories_stats(movies_id):
    total_categories = []
    for i in movies_id:
        cat_splitted = categories[i][0].split(",")
        cat_stripped = [cat.strip(' ') for cat in cat_splitted]
        total_categories.extend(cat_stripped)
    labels, freq = np.unique(total_categories, return_counts=True)
    idx = np.argsort(-freq)
    cat_sorted = [(labels[i],freq[i]) for i in idx]
    print(cat_sorted)


# In[ ]:


# 2.
def print_cluster_neurons_movies_report(cl_num):
    # create bmus array for the cluster (cl_bmus)
    rows, cols = np.where(som.clusters == cl_num)
    cl_bmus = np.column_stack((cols,rows))
    # check if bmu in ubmus and add neuron index to list
    neurons_idx_list = []
    for i in range (len(rows)):
        if 1 in np.all(ubmus==cl_bmus[i],axis=1):
            neurons_idx_list.append(np.where(np.all(ubmus==cl_bmus[i],axis=1))[0][0])
    print("Overall Cluster Genres stats:")
    neuron_movies_report(neurons_idx_list)


# In[ ]:


# 3.
def neuron_movies_report(neurons_list):
    movies_list = []
    for i in range(len(neurons_list)):
        mov_ids = [j for j,x in enumerate(indices) if x == neurons_list[i]]
        movies_list.extend(mov_ids)
    print_categories_stats(movies_list)


# In[ ]:


print_cluster_neurons_movies_report(1)


# 
# ## Tips για το SOM και το clustering
# 
# - Για την ομαδοποίηση ένα U-matrix καλό είναι να εμφανίζει και μπλε-πράσινες περιοχές (clusters) και κόκκινες περιοχές (ορίων). Παρατηρήστε ποια σχέση υπάρχει μεταξύ αριθμού ταινιών στο final set, μεγέθους grid και ποιότητας U-matrix.
# - Για το k του k-Means προσπαθήστε να προσεγγίζει σχετικά τα clusters του U-matrix (όπως είπαμε είναι διαφορετικοί μέθοδοι clustering). Μικρός αριθμός k δεν θα σέβεται τα όρια. Μεγάλος αριθμός θα δημιουργεί υπο-clusters εντός των clusters που φαίνονται στο U-matrix. Το τελευταίο δεν είναι απαραίτητα κακό, αλλά μεγαλώνει τον αριθμό clusters που πρέπει να αναλυθούν σημασιολογικά.
# - Σε μικρούς χάρτες και με μικρά final sets δοκιμάστε διαφορετικές παραμέτρους για την εκπαίδευση του SOM. Σημειώστε τυχόν παραμέτρους που επηρεάζουν την ποιότητα του clustering για το dataset σας ώστε να τις εφαρμόσετε στους μεγάλους χάρτες.
# - Κάποια τοπολογικά χαρακτηριστικά εμφανίζονται ήδη σε μικρούς χάρτες. Κάποια άλλα χρειάζονται μεγαλύτερους χάρτες. Δοκιμάστε μεγέθη 20x20, 25x25 ή και 30x30 και αντίστοιχη προσαρμογή των k. Όσο μεγαλώνουν οι χάρτες, μεγαλώνει η ανάλυση του χάρτη αλλά μεγαλώνει και ο αριθμός clusters που πρέπει να αναλυθούν.
# 

# 
# 
# ## Ανάλυση τοπολογικών ιδιοτήτων χάρτη SOM
# 
# Μετά το πέρας της εκπαίδευσης και του clustering θα έχετε ένα χάρτη με τοπολογικές ιδιότητες ως προς τα είδη των ταίνιών της συλλογής σας, κάτι αντίστοιχο με την εικόνα στην αρχή της Εφαρμογής 2 αυτού του notebook (η συγκεκριμένη εικόνα είναι μόνο για εικονογράφιση, δεν έχει καμία σχέση με τη συλλογή δεδομένων και τις κατηγορίες μας).
# 
# Για τον τελικό χάρτη SOM που θα παράξετε για τη συλλογή σας, αναλύστε σε markdown με συγκεκριμένη αναφορά σε αριθμούς clusters και τη σημασιολογική ερμηνεία τους τις εξής τρεις τοπολογικές ιδιότητες του SOM: 
# 
# 1. Δεδομένα που έχουν μεγαλύτερη πυκνότητα πιθανότητας στο χώρο εισόδου τείνουν να απεικονίζονται με περισσότερους νευρώνες στο χώρο μειωμένης διαστατικότητας. Δώστε παραδείγματα από συχνές και λιγότερο συχνές κατηγορίες ταινιών. Χρησιμοποιήστε τις στατιστικές των κατηγοριών στη συλλογή σας και τον αριθμό κόμβων που χαρακτηρίζουν.
# 2. Μακρινά πρότυπα εισόδου τείνουν να απεικονίζονται απομακρυσμένα στο χάρτη. Υπάρχουν χαρακτηριστικές κατηγορίες ταινιών που ήδη από μικρούς χάρτες τείνουν να τοποθετούνται σε διαφορετικά ή απομονωμένα σημεία του χάρτη.
# 3. Κοντινά πρότυπα εισόδου τείνουν να απεικονίζονται κοντά στο χάρτη. Σε μεγάλους χάρτες εντοπίστε είδη ταινιών και κοντινά τους υποείδη.
# 
# Προφανώς τοποθέτηση σε 2 διαστάσεις που να σέβεται μια απόλυτη τοπολογία δεν είναι εφικτή, αφενός γιατί δεν υπάρχει κάποια απόλυτη εξ ορισμού για τα κινηματογραφικά είδη ακόμα και σε πολλές διαστάσεις, αφετέρου γιατί πραγματοποιούμε μείωση διαστατικότητας.
# 
# Εντοπίστε μεγάλα clusters και μικρά clusters που δεν έχουν σαφή χαρακτηριστικά. Εντοπίστε clusters συγκεκριμένων ειδών που μοιάζουν να μην έχουν τοπολογική συνάφεια με γύρω περιοχές. Προτείνετε πιθανές ερμηνείες.
# 
# 
# 
# Τέλος, εντοπίστε clusters που έχουν κατά την άποψή σας ιδιαίτερο ενδιαφέρον στη συλλογή της ομάδας σας (data exploration / discovery value) και σχολιάστε.
# 

# ### *Τοπολογικές ιδιότητες του SOM*
# 1. Συχνές κατηγορίες ταινιών: Κατηγορίες με μεγάλη συχνότητα είναι οι Drama (cluster id: 24, number of neurons: 92) και Comedy (cluster id: 20, number of neurons: 62).  <br>
# Λιγότερο συχνές κατηγορίες ταινιών: Ενώ μικρή συχνότητα (από τις κυριατχούσες σε cluster κατηγορίες) εμφανίζουν οι Silent Film (cluster id: 15, number of neurons: 19) και Black-and-white (cluster id: 29, number of neurons: 17)<br><br>
# 2. Κατηγορίες που τοποθετούνται σε διαφορετικά ή απομονωμένα σημεία του χάρτη: Εύκολα εντοπίσιμες είναι οι απομακρυσένες κατηγορίες μεταξύ τους Comedy (id 20) και Horror (id 27). Το cluster της πρώτης κατηγορίας βρίσκεται πάνω δεξιά, ενώ αυτό της δεύτερης κάτω δεξιά. Ακόμα απομακρυσμένες είναι οι κύριες ομάδες των κατηγοριών Drama (id 24, κέντρο προς τα κάτω) και Comedy (id 20, πάνω δεξιά). Βλέπουμε λοιπόν ότι κατηγορίες που είναι θεματικά μακρυά, έχουν κατά συνέπεια clusters που απεικονίζονται απομακρυσμένα μεταξύ τους.<br><br>
# 3. Κατηγορίες ταινιών και κοντινά τους υποείδη: Βλέπουμε ότι η ομάδα με κυρίαρχη κατηγορία Adventure (id 16) γειτονεύει με ομάδες με κυρίαρχη κατηγορία την Action (id 10/26). Ένα ακόμα παράδειγμα είναι η γειτνίαση της ομάδας με κατηγορία Horror (id 27) με ομάδες με κατηγορία Thriller (id 3/6/17).<br>
# ### *Περαιτέρω εξερεύνηση*
# <br>Clusters που δεν έχουν σαφή χαρακτηριστικά: Ένα μικρό τέτοιο cluster (14 νευρώνες) είναι αυτό με id 9. Πιο συγκεκριμένα περιέχει στους top 3 counters τις εξής κατηγορίες: ('"Drama"', 51), ('"Romance Film"', 51), ('"Comedy"', 49). Δηλαδή κατηγορίες θεματικά πολύ μακρυά όπως Drama και Comedy με σχεδόν ίσες συχνότητες. <br>
# Ένα ακόμα τέτοιο cluster, αλλά μεγάλο αυτή τη φορά (28 νευρώνες), είναι αυτό με id 13. Οι top 2 κατηγορίες είναι οι εξής: ('"Comedy"', 136), ('"Drama"', 135). Παρατηρούμε τις ίδιες κατηγορίες με πριν σε σχεδόν ίσες πάλι συχνότητες.<br>
# Πιθανή ερμηνεία για ομάδες σαν κι αυτές είναι ότι βρίσκονται σε περιοχές ορίων (κόκκινες στην απεικόνιση του u-matrix) και έτσι δεν αποτελούν πραγματικά clusters, παρά μόνο διαχωριστικές περιοχές.<br><br>
# Clusters με ιδιαίτερο ενδιαφέρον για τη συλλογή μας είναι τα πιο ευδιάκριτα (τα πιο μπλε στην απεικόνιση του u-matrix) διότι σε αυτά βρίσκονται ταινίες με κοντινή θεματική ενότητα (και αντίστοιχα μεγάλους μετρητές παρόμοιων κατηγοριών και μικρούς μετρητές των υπόλοιπων/άσχετων κατηγοριών), οι οποίες μας είναι χρήσιμες για το σύστημα συστάσεων.<br>
# Clusters σαν αυτά είναι τα 24 (κατηγορία Drama), 20 (κατηγορία Comedy) και 1 (κατηγορία Romance Film)

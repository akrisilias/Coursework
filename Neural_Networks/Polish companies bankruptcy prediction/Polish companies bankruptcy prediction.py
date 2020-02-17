#!/usr/bin/env python
# coding: utf-8

# #Α. Στοιχεία ομάδας

# ##1.

# **Ομάδα Α9**
# 
# Μαύρος Γεώργιος,    03112618
# 
# Κρίσιλιας Ανδρέας,   03114778

# # Β. Εισαγωγή του dataset
# 

# ##1.

# Το dataset έχει να κάνει με την πρόβλεψη χρεοκοπίας Πολωνικών εταιριών. Περιέχει 64 οικονομικούς βαθμούς (attributes), 1 id και 1 πεδίο class για κάθε εταιρία - δείγμα. 
# 
# Τα attributes κάθε δείγματος έχουν καταγραφεί για 1 από τα 5 συνεχόμενα έτη κατά τα οποία έλαβε χώρα η συνολική καταγραφή.
# 
# Η στήλη class υποδεικνύει την κατάσταση της κάθε εταιρίας στο έκτο έτος. Και λαμβάνει τιμές: 0/μη χρεωκοπημένη, 1/χρεωκοπημένη.
# 
# 

# ##2.
# 

# Ανεβάζουμε το data.csv αρχείο που περιέχει τα δείγματα.

# In[ ]:


from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(name = fn, length = len(uploaded[fn])))


# Ενημέρωση βιβλιοθηκών

# In[ ]:


get_ipython().system(u'pip uninstall -y scikit-learn')
get_ipython().system(u'pip uninstall -y numpy')
get_ipython().system(u'pip uninstall -y pandas')

get_ipython().system(u'pip install scikit-learn')
get_ipython().system(u'pip install numpy')
get_ipython().system(u'pip install pandas')

import warnings 
warnings.filterwarnings('ignore')

get_ipython().system(u'pip uninstall -y imbalanced-learn')
get_ipython().system(u'pip install imbalanced-learn')


# Ελέγχουμε ότι το αρχείο που ανεβάσαμε υπάρχει στο file system.

# In[ ]:


get_ipython().system(u'ls')


# Διαβάζουμε το αρχείο (αντικαθιστώντας ταυτόχρονα τις τιμές '?' με NaN) και τυπώνουμε τις 5 πρώτες γραμμές.

# In[ ]:


import pandas as pd
from io import StringIO

# load dataset as pandas.DataFrame
df = pd.read_csv("data.csv", na_values=["?"], header = None)
df.head()


# Αποθήκευση των attributes και της κλάσης σε διαφορετικά dataframes και μετατροπή αυτών σε numpy arrays.

# In[ ]:


# labels and features dataframes
labels_df = df.iloc[:,-1]
features_df = df.iloc[:,0:-1]

# convert to numpy arrays
np_labels = labels_df.values.flatten()
np_features = features_df.values


# Αριθμός δειγμάτων

# In[ ]:


# number of examples
np_labels.size


# Αριθμός χαρακτηριστικών

# In[ ]:


# number of features
np_features.shape[1]


# Είδος χαρακτηριστικών

# In[ ]:


# features data-type of first sample
features_df.dtypes


# Παρατηρούμε λοιπόν ότι όλα τα χαρακτηριστικά είναι τύπου float.
# 
# Δε φαίνονται βέβαια τα Αttr30 έως Attr33 οπότε τα τυπώνουμε κι αυτά από κάτω

# In[ ]:


# also print feautures 30-33 dtypes
for i in range (30,34):
  print(str(i) + " ", end = '')
  print(type(np_features[0,i]))


# Δεν έχουμε ούτε χαρακτηριστικά τύπου string, ούτε τύπου int (τα οποία θα μπορούσαν να υποδηλώνουν κάποια αντιστοιχία με κατηγορία). Επομένως συμπεραίνουμε ότι όλα τα χαρακτηριστικά μας είναι διατεταγμένα (τύπου float όλα).

# ##3.
# 

# Υπάρχει αρίθμηση γραμμών αλλά όχι επικεφαλίδες, όπως φαίνεται και από την τύπωση του dataframe παρακάτω.

# In[ ]:


print(df)


# ##4.

# Οι ετικέτες των κλάσεων παίρνουν τιμές 0 (για μη χρεωκοπία) και 1 (για χρεωκοπία). Βρίσκονται στην τελευταία (66η) κολόνα του dataframe, και είναι οι εξής:

# In[ ]:


# print labels
print(labels_df)


# ##5.
# To dataset αποτελείται από 5 αρχεία .arff (1year.arff, 2year.arff, 3year.arff, 4year.arff, 5year.arff) τα οποία μετατρέπουμε σε .csv μέσω του bash:
# 
# cat 1year.arff | grep -ve "^@\|^%" | grep -v "^[[:space:]]*$" > data1.csv
# 
# cat 2year.arff | grep -ve "^@\|^%" | grep -v "^[[:space:]]*$" > data2.csv
# 
# cat 3year.arff | grep -ve "^@\|^%" | grep -v "^[[:space:]]*$" > data3.csv
# 
# cat 4year.arff | grep -ve "^@\|^%" | grep -v "^[[:space:]]*$" > data4.csv
# 
# cat 5year.arff | grep -ve "^@\|^%" | grep -v "^[[:space:]]*$" > data5.csv
# <br><br>
# Και τα κάνουμε concatenate πάλι μέσω του bash:
# 
# cat \*.csv > data.csv
# <br><br>
# Και παίρνουμε έτσι το τελικό αρχείο: *data.csv*, στο οποίο αντικαθιστούμε κατά την ανάγνωση τους χαρακτήρες '?' με NaN. Αυτό το κάνουμε δίνοντας στη συνάρτηση read_csv ως παράμετρο na_values=["?"].
# 

# ##6.

# Εύρεση δειγμάτων με απουσιάζουσες τιμές.

# In[ ]:


# find examples with missing values
missing = sum([True for idx,row in df.iterrows() if any(row.isnull())])
missing


# Επομένως υπάρχουν απουσιάζουσες τιμές. 
# 
# Τα δείγματα με τουλάχιστον μία απουσιάζουσα τιμή είναι 23438 όπως φαίνεται και παραπάνω.

# Ποσοστό δειγμάτων με απουσιάζουσες τιμές

# In[ ]:


# examples with missing values percentage
print(missing / np_labels.size * 100, end = '')
print(" %")


# ##7.

# Υπάρουν 2 κλάσεις: 0 για μη χρεωκοπημένες εταιρίες και 1 για χρεωκοπημένες.

# In[ ]:


import numpy as np

print("Unique classes: ", np.unique(np_labels))

print("Number of classes: ", np.unique(np_labels).size)


# Τα ποσοστά των δειγμάτων τους επί του συνόλου φαίνεται παρακάτω:

# In[ ]:


print("Examples per class percentage (%): ", np.bincount(np_labels)/np_labels.size*100)


# Όπως εύκολα διακρίνουμε, το dataset μας δεν είναι καθόλου ισορροπημένο καθότι τα δείγματα της κλάσης 0 είναι πάνω από 19 φορές περισσότερα αυτών της κλάσης 1:
# 

# In[ ]:


freq = np.bincount(np_labels)
print("Class frequency ratio: ", freq[0]/freq[1])


# ##8.

# Όπως είδαμε παραπάνω το ποσοστό των δειγμάτων με απουσιάζουσες τιμές είναι σχεδόν 54%. Επομένως δεν υπάρχει περίπτωση να θυσιάσουμε τόσο πολλά δείγματα. Θα κάνουμε χρήση του μετασχηματιστή Imputer,  ώστε να αντικαταστήσουμε κάθε απουσιάζουσα τιμή με τη μέση τιμή του χαρακτηριστικού στο train set.

# Χωρίζουμε, λοιπόν, αρχικά το dataset σε train και test set σε αναλογία 70-30.

# In[ ]:


from sklearn.model_selection import train_test_split

# split data intro train and test sets
X_train, X_test, y_train, y_test = train_test_split(np_features, np_labels, test_size = 0.3)


# Και στη συνέχεια εφαρμόζουμε τον Imputer στο train set.

# In[ ]:


from sklearn.preprocessing import Imputer

# replace missing values with the mean value
imp = Imputer(strategy = 'mean', axis = 0)
imp = imp.fit(X_train)
X_train = imp.transform(X_train)


# Και τον εφαρμόζουμε επίσης και στο test set.

# In[ ]:


X_test = imp.transform(X_test)


# #Γ. Baseline classification

# ##1.

# Εκπαίδευση των ταξινομητών στο train set με απλή αρχικοποίηση (default τιμές).

# In[ ]:


# Baseline classification

# Initialization and training

# α) dummy classifiers
from sklearn.dummy import DummyClassifier

dc_uniform = DummyClassifier(strategy="uniform")
dc_constant_0 = DummyClassifier(strategy="constant", constant=0)
dc_constant_1 = DummyClassifier(strategy="constant", constant=1)
dc_most_frequent = DummyClassifier(strategy="most_frequent")
dc_stratified = DummyClassifier(strategy="stratified")

dc_uniform.fit(X_train, y_train)
dc_constant_0.fit(X_train, y_train)
dc_constant_1.fit(X_train, y_train)
dc_most_frequent.fit(X_train, y_train)
dc_stratified.fit(X_train, y_train)

# β) Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

# γ) kNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier() # default n_neighbors=5

knn.fit(X_train, y_train)

# δ) Multi-Layer Perceptron (MLP)
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()

mlp.fit(X_train, y_train)



# Κάνουμε εκτίμηση στο test set για τους παραπάνω ταξινομητές.

# In[ ]:


# Prediction

# α) dummy classifiers
dc_uni_pred = dc_uniform.predict(X_test)
dc_co0_pred = dc_constant_0.predict(X_test)
dc_co1_pred = dc_constant_1.predict(X_test)
dc_mos_pred = dc_most_frequent.predict(X_test)
dc_str_pred = dc_stratified.predict(X_test)

# β) Gaussian Naive Bayes
gnb_pred = gnb.predict(X_test)

# γ) kNN
knn_pred = knn.predict(X_test)

# δ) Multi-Layer Perceptron (MLP)
mlp_pred = mlp.predict(X_test)

# Estimation
from sklearn.metrics import accuracy_score

accuracy = {}

# α) dummy classifiers
accuracy['uniform'] = accuracy_score(y_test, dc_uni_pred)
accuracy['constant 0'] = accuracy_score(y_test, dc_co0_pred)
accuracy['constant 1'] = accuracy_score(y_test, dc_co1_pred)
accuracy['most frequent'] = accuracy_score(y_test, dc_mos_pred)
accuracy['stratified'] = accuracy_score(y_test, dc_str_pred)

# β) Gaussian Naive Bayes
accuracy['gaussian naive bayes'] = accuracy_score(y_test, gnb_pred)

# γ) kNN
accuracy['kNN'] = accuracy_score(y_test, knn_pred)

# δ) Multi-Layer Perceptron (MLP)
accuracy['mlp'] = accuracy_score(y_test, mlp_pred)

# results
print("Baseline classification accurary\n")
sorted_accuracy = [(k, accuracy[k]) for k in sorted(accuracy, key=accuracy.get, reverse=True)]
for k, v in sorted_accuracy:
  print(k, v)


# Παρατηρούμε ότι λόγω της μεγάλης ανισορροπίας του dataset κάποιοι dummy classifiers πέτυχαν την καλύτερη πιστότητα, ενώ αντιθέτως ο Gaussian Naive Bayes είχε πιστότητα κάτω από 7,1%.

# Τυπώνουμε για κάθε estimator: confusion matrix, f1-micro average και f1-macro average.

# In[ ]:


# define the label names
label_names = ['Not Bankrupt', 'Bankrupt' ]

# confusion matrix, f1-micro average, f1-macro average
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

metrics_names = ['precision', 'recall', 'f1-score', 'support']

# function to print confusion matrix, f1-micro average, f1-macro average
def cm_mic_mac_function(name, pred):
  print("For ", name, " we've got the following:\n")
  cm = confusion_matrix(y_test, pred)
  print("Confusion Matrix")
  print(label_names)
  print(cm, "\n")
  
  print("micro avg")
  print(metrics_names)
  mic = precision_recall_fscore_support(y_test, pred, average='micro')
  print(mic)
  print("f1 micro average = ", mic[2], "\n")
  print("macro avg")
  print(metrics_names)
  mac = precision_recall_fscore_support(y_test, pred, average='macro')
  print(mac)
  print("f1 macro average = ", mac[2], "\n\n\n")
  return mic, mac 

# α) dummy classifiers             
(uni_pr_mic,uni_re_mic,uni_f1_mic,_), (uni_pr_mac,uni_re_mac,uni_f1_mac,_) = cm_mic_mac_function('dc uniform', dc_uni_pred)
(co0_pr_mic,co0_re_mic,co0_f1_mic,_), (co0_pr_mac,co0_re_mac,co0_f1_mac,_) = cm_mic_mac_function('dc constant 0', dc_co0_pred)
(co1_pr_mic,co1_re_mic,co1_f1_mic,_), (co1_pr_mac,co1_re_mac,co1_f1_mac,_) = cm_mic_mac_function('dc constant 1', dc_co1_pred)
(mos_pr_mic,mos_re_mic,mos_f1_mic,_), (mos_pr_mac,mos_re_mac,mos_f1_mac,_) = cm_mic_mac_function('dc most frequent', dc_mos_pred)
(str_pr_mic,str_re_mic,str_f1_mic,_), (str_pr_mac,str_re_mac,str_f1_mac,_) = cm_mic_mac_function('dc stratified', dc_str_pred)

# β) Gaussian Naive Bayes
(gnb_pr_mic,gnb_re_mic,gnb_f1_mic,_), (gnb_pr_mac,gnb_re_mac,gnb_f1_mac,_) = cm_mic_mac_function('Gaussian Naive Bayes', gnb_pred)

# γ) kNN
(knn_pr_mic,knn_re_mic,knn_f1_mic,_), (knn_pr_mac,knn_re_mac,knn_f1_mac,_) = cm_mic_mac_function('kNN', knn_pred)

# δ) Multi-Layer Perceptron (MLP)
(mlp_pr_mic,mlp_re_mic,mlp_f1_mic,_), (mlp_pr_mac,mlp_re_mac,mlp_f1_mac,_) = cm_mic_mac_function('Multi-Layer Perceptron (MLP)', mlp_pred)




# ##2.

# Αρχικά θα ορίσουμε μία συνάρτηση μέσω της οποίας θα εμφανίζονται τα διαγράμματα.

# In[ ]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

objects = ('Uni', 'Co0', 'Co1', 'MoF', 'Str', 'GNB', 'kNN', 'MLP')
y_pos = np.arange(len(objects))

# function to print barplots
def bar_plot_function(title, performance):
  plt.bar(y_pos, performance, align='center', alpha=0.5)
  plt.xticks(y_pos, objects)
  plt.ylabel('Value')
  plt.title(title)
  plt.show()


# Για f1 micro average έχουμε το παρακάτω plot:
# 

# In[ ]:


perf = [uni_f1_mic, co0_f1_mic, co1_f1_mic, mos_f1_mic, str_f1_mic, gnb_f1_mic, knn_f1_mic, mlp_f1_mic]
bar_plot_function('Average Micro F1-Score', perf)


# Για f1 macro average έχουμε τα παρακάτω plots:

# In[ ]:


perf = [uni_f1_mac, co0_f1_mac, co1_f1_mac, mos_f1_mac, str_f1_mac, gnb_f1_mac, knn_f1_mac, mlp_f1_mac]
bar_plot_function('Average Macro F1-Score', perf)


# ##3.

# Ο DC uniform βλέπουμε ότι σε όλες τις μετρικές του έχει ποσοστά γύρω στο 0.5, τόσο σε micro όσο και σε macro average, πράγμα λογικό αφού κατατάσσει σωστά για κάθε κλάση τα μισά περίπου δείγματα.<br><br>
# Ο DC constant 0 προβλέπει το 95% περίπου των δειγμάτων χάρη στην τεράστια ανισορροπία του dataset. Έτσι πετυχαίνει πολύ καλές μετρικές για micro average. Στα macro average, όμως, οι μετρικές του πέφτουνε περίπου στη μέση. Αυτό συμβαίνει διότι σε αυτήν την περίπτωση βγαίνει ο μέσος όρος των μετρικών των 2 κλάσεων, και ενώ τα έχει πάει καλά για την κλάση 0, στην 1 δεν κατατάξει τίποτα και έτσι ο μέσος όρος πάει κάπου στο 0.5.<br><br>
# Για αντίστοιο λόγο ο DC constant 1 δεν έχει καλές μετρικές ούτε σε micro ούτε σε macro average.<br><br>
# Ο DC most frequent μαντεύει πάντα 0, οπότε έχει ακριβώς ίδες μετρικές με τον constant 0.<br><br>
# Ο DC stratified λόγω του ότι προσπαθεί να μαντέψει και δείγματα της κλάσης 1 με ένα μικρό ποσοστό (αντίστοιχο της ισορροπίας των κλάσεων) έχει χειρότερες μετρικές σε micro average, αλλά καλύτερες εν τέλει σε macro.<br><br>
# O Gaussian Naive Bayes βλέπουμε ότι πετυχένει εν γένει πολύ χαμηλή απόδοση. Πιθανώς για αυτό να ευθύνεται η μεγάλη ανισορροπία του dataset και η έλλειψη κανονικοποίησης.<br><br>
# Οι kNN και MLP βλέπουμε ότι έχουν γενικά καλές μετρικές τόσο σε micro average, όπου βρίσκονται κοντά στον most frequent, παρά τη μεγάλη ανισορροπία, όσο και σε macro average, όπου ξεπερνάνε την απόδοση των dummies.

# #Δ. Βελτιστοποίηση ταξινομητών

# ##1, 2.

# Θα βελτιστοποιήσουμε την απόδοση του κάθε ταξινομητή μέσω προεπεξεργασίας και εύρεσης βέλτιστων υπερπαραμέτρων (για όσους έχουν υπερπαραμέτρους).<br>
# Οι παράμετροι που θα προσπαθήσουμε βελτιστοποιήσουμε για τους μετασχηματιστές είναι:<br>
# *threshold --> για τον VarianceThreshold<br>
# n_components --> για τον PCA*<br>
# Θα χρησιμοποιηθεί λογική bottom-up, δηλαδή αρχικά θα χρησιμοποιούμε όλους τους διαθέσιμους transformers και στη συνέχεια θα κάνουμε δοκιμές χωρίς κάποιους από αυτούς.

# Αρχικοποίηση μετασχηματιστών και imports

# In[ ]:


from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA

from sklearn.metrics import classification_report

selector = VarianceThreshold()
scaler = StandardScaler()
ros = RandomOverSampler()
pca = PCA()


# Μελέτη του variance των μεταβλητών για τη VarianceThreshold

# In[ ]:


# variance analysis
train_variance = X_train.var(axis=0)
print(train_variance)
print("The maximum variance is: ", np.max(train_variance))
print("The minimum variance is: ", np.min(train_variance))
print("The mean variance is: ", np.mean(train_variance))


# Επομένως θα χρησιμοποιήσουμε ως πιθανές τιμές παραμέτρων για τους transformers τις παρακάτω.

# In[ ]:


vthreshold = [0, 1, 3, 5, 7, 10, 100, 1000, 10000]
n_components = [1, 2, 4, 6, 9, 12]


# Ορίζουμε συνάρτηση για εμφάνιση του Covariance Matrix:

# In[ ]:


# function to print confusion matrix
def print_cm(pred):
  cm = confusion_matrix(y_test, pred)
  print("Confusion Matrix")
  print(label_names)
  print(cm, "\n")


# **Dummy Classifiers**

# Οι μετατροπές που αφορούν τα χαρακτηριστικά (επιλογή, κανονικοποίηση, εξαγωγή νέων) δεν επηρεάζουν τους Dummy ταξινομητές, αφού κανένας απ' αυτούς δεν εξετάζει τα χαρακτηριστικά είτε του train είτε του test set.<br>
# Ο μόνος από τους transformers που μπορεί να επηρεάσει είναι ο RandomOverSampler. Από αυτόν επηρεάζεται σίγουρα ο stratified, αφού λαμβάνει υπόψιν του τη νέα ισορροπία, η οποία όμως τον οδηγεί σε εσφαλμένα συμπεράσματα και έτσι πέφτει η γενική του απόδοση. Θα μπορούσε ακόμα να επηρεαστεί ο Most Frequent σε περίπτωση που μέσω της υπερδειγματοληψίας άλλαζε η κλάση με τα περισσότερα δείγματα στο train set. Αλλά αυτό θα είχε επίσης αρνητική επίπτωση στην απόδοση του ταξινομητή, αφού θα προέβλεπε συνεχώς τη λιγότερο συχνή κλάση.

# **Gaussian Naive Bayes**

# Τυπώνουμε αρχικά τις μετρικές του estimator χωρίς το pipeline.

# In[ ]:


print("Accuracy: ", accuracy_score(y_test, gnb_pred))
print("\nMetrics: \n", classification_report(y_test, gnb_pred))


# ***f1-micro***

# Δοκιμάζουμε αρχικά ένα pipeline με όλους τους transformers που ορίσαμε.

# In[ ]:


pipe_gnb = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('gnb', gnb)])
gnb_estimator1 = GridSearchCV(pipe_gnb, dict(selector__threshold=vthreshold, pca__n_components=n_components), scoring='f1_micro', n_jobs=-1)
gnb_estimator1.fit(X_train, y_train)
gnb_est1_pred = gnb_estimator1.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, gnb_est1_pred))
print("\nMetrics: \n", classification_report(y_test, gnb_est1_pred))
print("Optimum hyperparameters: ", gnb_estimator1.best_params_)


# ('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('gnb', gnb)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'pca__n_components': 1, 'selector__threshold': 1000}<br>* αλλάζουν οι τιμές όταν κάνουμε πολλά run (ισχύει και παρακάτω)

# Και δοκιμάζουμε να αφαιρέσουμε καθέναν ξεχωριστά.<br>
# Παρατηρούμε ότι αν αφαιρέσουμε κάποιον εκτός του selector, οι μετρικές πέφτουν δραματικά. Ενώ χωρίς το selector παραμένουν σε παρόμοια επίπεδα. Επομένως θα κρατήσουμε όλους τους μετασχηματιστές.

# Με αφαίρεση μετασχηματιστών ανά 2, επίσης δε βλέπουμε κάποια βελτίωση όσον αφορά την f1-micro.

# Θα εξετάσουμε τώρα διαστήματα των υπερπαραμέτρων πιο κοντά στις τιμές που βρήκαμε.

# In[ ]:


vthreshold1 = [0, 1, 2, 3]
n_components1 = [1, 2, 3, 4]

pipe_gnb = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('gnb', gnb)])
gnb_estimator1 = GridSearchCV(pipe_gnb, dict(selector__threshold=vthreshold1, pca__n_components=n_components1), scoring='f1_micro', n_jobs=-1)
gnb_estimator1.fit(X_train, y_train)
gnb_est1_pred = gnb_estimator1.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, gnb_est1_pred))
print("\nMetrics: \n", classification_report(y_test, gnb_est1_pred))
print("Optimum hyperparameters: ", gnb_estimator1.best_params_)


# ('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('gnb', gnb)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'pca__n_components': 1, 'selector__threshold': 3}

# Άρα έχουμε πάλι την ίδια τιμή f1-micro.

# Ορίζουμε λοιπόν τον τελικό estimator, και τυπώνουμε τις f1 μετρικές του και το accuracy.

# In[ ]:


import time

selector1 = VarianceThreshold(threshold=3)
scaler1 = StandardScaler()
ros1 = RandomOverSampler()
pca1 = PCA(n_components=1)
gnb1 = GaussianNB()

Gnb_mic = Pipeline(steps=[('selector', selector1), ('scaler', scaler1), ('sampler', ros1), ('pca', pca1), ('gnb', gnb1)])

time1 = time.time()
Gnb_mic.fit(X_train, y_train)
time2 = time.time()
Gnb_mic_pred = Gnb_mic.predict(X_test)
time3 = time.time()

_,_,Gnb_f1_mic,_ = precision_recall_fscore_support(y_test, Gnb_mic_pred, average='micro')
_,_,f1_mac,_ = precision_recall_fscore_support(y_test, Gnb_mic_pred, average='macro')

print("f1-micro: ", Gnb_f1_mic)
print("f1-macro: ", f1_mac)
print("Accuracy: ", accuracy_score(y_test, Gnb_mic_pred))



# Και έχουμε τον εξής Confusion Matrix:

# In[ ]:


print_cm(Gnb_mic_pred)


# Χρόνοι εκτέλεσης

# In[ ]:


train_time = time2 - time1
test_time = time3 - time2
print("train time: ", train_time)
print("test time: ", test_time)


# ***f1-macro***

# Αντίστοιχα τώρα για f1-macro

# In[ ]:


pipe_gnb = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('gnb', gnb)])
gnb_estimator2 = GridSearchCV(pipe_gnb, dict(selector__threshold=vthreshold, pca__n_components=n_components), scoring='f1_macro', n_jobs=-1)
gnb_estimator2.fit(X_train, y_train)
gnb_est2_pred = gnb_estimator2.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, gnb_est2_pred))
print("\nMetrics: \n", classification_report(y_test, gnb_est2_pred))
print("Optimum hyperparameters: ", gnb_estimator2.best_params_)


# ('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('gnb', gnb)<br>0.49<br>Βέλτιστοι υπερπαράμετροι:  {'pca__n_components': 6, 'selector__threshold': 1}

# Ούτε πάλι γίνεται κάποια θετική αλλαγή με αφαιρέσεις μετασχηματιστών.

# Επαναλαμβάνουμε ένα GridSearch σε πιο κοντινό διάστημα.

# In[ ]:


vthreshold3 = [0, 1, 2]
n_components3 = [2, 3, 4, 5, 6]

pipe_gnb = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('gnb', gnb)])
gnb_estimator2 = GridSearchCV(pipe_gnb, dict(selector__threshold=vthreshold3, pca__n_components=n_components3), scoring='f1_macro', n_jobs=-1)
gnb_estimator2.fit(X_train, y_train)
gnb_est2_pred = gnb_estimator2.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, gnb_est2_pred))
print("\nMetrics: \n", classification_report(y_test, gnb_est2_pred))
print("Optimum hyperparameters: ", gnb_estimator2.best_params_)


# ('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('gnb', gnb)<br>0.49<br>Βέλτιστοι υπερπαράμετροι:  {'pca__n_components': 6, 'selector__threshold': 2}

# Και βρίσκουμε πάλι ίδια τιμή για f1-macro.

# Ορίζουμε λοιπόν τον τελικό estimator, και τυπώνουμε τις f1 μετρικές του και το accuracy.

# In[ ]:


selector1 = VarianceThreshold(threshold=2)
scaler1 = StandardScaler()
ros1 = RandomOverSampler()
pca1 = PCA(n_components=6)
gnb1 = GaussianNB()

Gnb_mac = Pipeline(steps=[('selector', selector1), ('scaler', scaler1), ('sampler', ros1), ('pca', pca1), ('gnb', gnb1)])

time1 = time.time()
Gnb_mac.fit(X_train, y_train)
time2 = time.time()
Gnb_mac_pred = Gnb_mac.predict(X_test)
time3 = time.time()

_,_,f1_mic,_ = precision_recall_fscore_support(y_test, Gnb_mac_pred, average='micro')
_,_,Gnb_f1_mac,_ = precision_recall_fscore_support(y_test, Gnb_mac_pred, average='macro')

print("f1-micro: ", f1_mic)
print("f1-macro: ", Gnb_f1_mac)
print("Accuracy: ", accuracy_score(y_test, Gnb_mac_pred))



# Και έχουμε τον εξής Confusion Matrix:

# In[ ]:


print_cm(Gnb_mac_pred)


# Χρόνοι εκτέλεσης

# In[ ]:


train_time = time2 - time1
test_time = time3 - time2
print("train time: ", train_time)
print("test time: ", test_time)


# **kNN**

# Ορίζουμε από την αρχή τις υπερπαραμέτρους.

# In[ ]:


vthreshold_ = [5, 1000]
n_components_ = [3, 15]
n_neighbors = [1, 10]
metric = ['euclidean', 'manhattan']
weights = ['uniform', 'distance']


# Τυπώνουμε αρχικά τις μετρικές του estimator χωρίς το pipeline.

# In[ ]:


print("Accuracy: ", accuracy_score(y_test, knn_pred))
print("\nMetrics: \n", classification_report(y_test, knn_pred))


# ***f1-micro***

# Θα δοκιμάσουμε όλους τους δυνατούς συνδυασμούς για τους μετασχηματιστές VarianceThreshold - StandrardScaler - ROS - PCA.

# Θα χρησιμοποιήσουμε τον παρακάτω κώδικα, προσθαφαιρώντας κάθε φορά μετασχηματιστές (μαζί με τις αντίστοιχες παραμέτρους τους).

# In[ ]:


#pipe_knn = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('knn', knn)], memory = 'tmp')
#knn_estimator1 = GridSearchCV(pipe_knn, dict(selector__threshold=vthreshold_, pca__n_components=n_components_, knn__n_neighbors=n_neighbors, knn__metric=metric, knn__weights=weights), cv=5, scoring='f1_micro', n_jobs=-1)
#knn_estimator1.fit(X_train, y_train)
#knn_est1_pred = knn_estimator1.predict(X_test)
#print("Accuracy: ", accuracy_score(y_test, knn_est1_pred))
#print("\nMetrics: \n", classification_report(y_test, knn_est1_pred))
#print("Optimum hyperparameters: ", knn_estimator1.best_params_)


# Παίρνουμε τα κάτωθι αποτελέσματα:

# ('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('knn', knn)
# <br>f1-micro = 0.91<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'pca__n_components': 15, 'selector__threshold': 5}
# 

# ('scaler', scaler), ('sampler', ros), ('pca', pca), ('knn', knn)<br>f1-micro = 0.92<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'pca__n_components': 15}

# ('selector', selector), ('sampler', ros), ('pca', pca), ('knn', knn)<br>f1-micro = 0.92<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'pca__n_components': 15, 'selector__threshold': 5}
# 
# 

# ('selector', selector), ('scaler', scaler), ('pca', pca), ('knn', knn)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 10, 'knn__weights': 'uniform', 'pca__n_components': 15, 'selector__threshold': 1000}

# ('selector', selector), ('scaler', scaler), ('sampler', ros), ('knn', knn)<br>f1-micro = 0.93<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'selector__threshold': 1000}

# ('sampler', ros), ('pca', pca), ('knn', knn)<br>f1-micro = 0.92<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'pca__n_components': 15}

# ('scaler', scaler), ('pca', pca), ('knn', knn)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'euclidean', 'knn__n_neighbors': 10, 'knn__weights': 'uniform', 'pca__n_components': 3}

# ('scaler', scaler), ('sampler', ros), ('knn', knn)<br>f1-micro = 0.93<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform'}

# ('selector', selector), ('pca', pca), ('knn', knn)
# <br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 10, 'knn__weights': 'uniform', 'pca__n_components': 15, 'selector__threshold': 5}
# 
# 

# ('selector', selector), ('sampler', ros), ('knn', knn)]<br>f1-micro = 0.92<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'euclidean', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'selector__threshold': 1000}

# ***('selector', selector), ('scaler', scaler), ('knn', knn)<br>f1-micro = 0.96<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'euclidean', 'knn__n_neighbors': 10, 'knn__weights': 'uniform', 'selector__threshold': 1000}***

# ('selector', selector), ('knn', knn)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 10, 'knn__weights': 'uniform', 'selector__threshold': 1000}

# ('scaler', scaler), ('knn', knn)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 10, 'knn__weights': 'uniform'}

# ('sampler', ros), ('knn', knn)<br>f1-micro = 0.92<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'euclidean', 'knn__n_neighbors': 1, 'knn__weights': 'uniform'}

# ('pca', pca), ('knn', knn)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 10, 'knn__weights': 'uniform', 'pca__n_components': 15}

# ('knn', knn)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 10, 'knn__weights': 'uniform'}

# Ορίζουμε λοιπόν τον τελικό estimator, και τυπώνουμε τις f1 μετρικές του και το accuracy.

# In[ ]:


selector1 = VarianceThreshold(threshold=1000)
scaler1 = StandardScaler()
knn1 = KNeighborsClassifier(metric='euclidean', n_neighbors=10, weights='uniform')

Knn_mic = Pipeline(steps=[('selector', selector1), ('scaler', scaler1), ('knn', knn1)])

time1 = time.time()
Knn_mic.fit(X_train, y_train)
time2 = time.time()
Knn_mic_pred = Knn_mic.predict(X_test)
time3 = time.time()

_,_,Knn_f1_mic,_ = precision_recall_fscore_support(y_test, Knn_mic_pred, average='micro')
_,_,f1_mac,_ = precision_recall_fscore_support(y_test, Knn_mic_pred, average='macro')

print("f1-micro: ", Knn_f1_mic)
print("f1-macro: ", f1_mac)
print("Accuracy: ", accuracy_score(y_test, Knn_mic_pred))



# Και έχουμε τον εξής Confusion Matrix:

# In[ ]:


print_cm(Knn_mic_pred)


# Χρόνοι εκτέλεσης

# In[ ]:


train_time = time2 - time1
test_time = time3 - time2
print("train time: ", train_time)
print("test time: ", test_time)


# ***f1-macro***

# Θα δοκιμάσουμε όλους τους δυνατούς συνδυασμούς για τους μετασχηματιστές VarianceThreshold - StandrardScaler - ROS - PCA.

# Θα χρησιμοποιήσουμε τον παρακάτω κώδικα, προσθαφαιρώντας κάθε φορά μετασχηματιστές (μαζί με τις αντίστοιχες παραμέτρους τους).

# In[ ]:


#pipe_knn = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('knn', knn)], memory = 'tmp')
#knn_estimator2 = GridSearchCV(pipe_knn, dict(selector__threshold=vthreshold_, pca__n_components=n_components_, knn__n_neighbors=n_neighbors, knn__metric=metric, knn__weights=weights), cv=5, scoring='f1_macro', n_jobs=-1)
#knn_estimator2.fit(X_train, y_train)
#knn_est2_pred = knn_estimator2.predict(X_test)
#print("Accuracy: ", accuracy_score(y_test, knn_est2_pred))
#print("\nMetrics: \n", classification_report(y_test, knn_est2_pred))
#print("Optimum hyperparameters: ", knn_estimator2.best_params_)


# Παίρνουμε τα κάτωθι αποτελέσματα:

# ('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('knn', knn)<br>f1-macro = 0.55<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'pca__n_components': 15, 'selector__threshold': 1000}
# 

# ('scaler', scaler), ('sampler', ros), ('pca', pca), ('knn', knn)<br>f1-macro = 0.54<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'pca__n_components': 15}

# ('selector', selector), ('sampler', ros), ('pca', pca), ('knn', knn)<br>f1-macro = 0.57<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'pca__n_components': 15, 'selector__threshold': 5}

# ('selector', selector), ('scaler', scaler), ('pca', pca), ('knn', knn)<br>f1-macro = 0.61<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'euclidean', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'pca__n_components': 15, 'selector__threshold': 1000}

# ('selector', selector), ('scaler', scaler), ('sampler', ros), ('knn', knn)<br>f1-macro = 0.62<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'euclidean', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'selector__threshold': 1000}

# ('sampler', ros), ('pca', pca), ('knn', knn)<br>f1-macro = 0.57<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'pca__n_components': 15}

# ('scaler', scaler), ('pca', pca), ('knn', knn)<br>f1-macro = 0.55<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'pca__n_components': 15}

# ('scaler', scaler), ('sampler', ros), ('knn', knn)<br>f1-macro = 0.61<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform'}

# ('selector', selector), ('pca', pca), ('knn', knn)<br>f1-macro = 0,57<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'pca__n_components': 15, 'selector__threshold': 1000}

# ('selector', selector), ('sampler', ros), ('knn', knn)<br>f1-macro = 0.58<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'selector__threshold': 5}

# ***('selector', selector), ('scaler', scaler), ('knn', knn)<br>f1-macro = 0.62<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'euclidean', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'selector__threshold': 1000}***

# ('selector', selector), ('knn', knn)<br>f1-macro = 0.57<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'selector__threshold': 1000}

# ('scaler', scaler), ('knn', knn)<br>f1-macro = 0.60<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform'}

# ('sampler', ros), ('knn', knn)<br>f1-macro = 0.58<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform'}

# ('pca', pca), ('knn', knn)<br>f1-macro = 0.58<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'euclidean', 'knn__n_neighbors': 1, 'knn__weights': 'uniform', 'pca__n_components': 15}

# ('knn', knn)<br>f1-macro = 0.58<br>Βέλτιστοι υπερπαράμετροι:  {'knn__metric': 'manhattan', 'knn__n_neighbors': 1, 'knn__weights': 'uniform'} 

# Ορίζουμε λοιπόν τον τελικό estimator, και τυπώνουμε τις f1 μετρικές του και το accuracy.

# In[ ]:


selector1 = VarianceThreshold(threshold=1000)
scaler1 = StandardScaler()
knn1 = KNeighborsClassifier(metric='manhattan', n_neighbors=1, weights='uniform')

Knn_mac = Pipeline(steps=[('selector', selector1), ('scaler', scaler1), ('knn', knn1)])

time1 = time.time()
Knn_mac.fit(X_train, y_train)
time2 = time.time()
Knn_mac_pred = Knn_mac.predict(X_test)
time3 = time.time()

_,_,f1_mic,_ = precision_recall_fscore_support(y_test, Knn_mac_pred, average='micro')
_,_,Knn_f1_mac,_ = precision_recall_fscore_support(y_test, Knn_mac_pred, average='macro')

print("f1-micro: ", f1_mic)
print("f1-macro: ", Knn_f1_mac)
print("Accuracy: ", accuracy_score(y_test, Knn_mac_pred))



# Και έχουμε τον εξής Confusion Matrix:

# In[ ]:


print_cm(Knn_mac_pred)


# Χρόνοι εκτέλεσης

# In[ ]:


train_time = time2 - time1
test_time = time3 - time2
print("train time: ", train_time)
print("test time: ", test_time)


# **MLP**

# Ορίζουμε τις υπερπαραμέτρους για τον MLP.

# In[ ]:


vthreshold_ = [5, 1000]
n_components_ = [3, 15]
hidden_layer_sizes = [(50,), (100,)] # 100,
activation = ['relu', 'tanh'] # relu
solver = ['adam', 'sgd'] # 'adam' works better with large datasets
max_iter = [50, 200] # 200
learning_rate = ['constant', 'invscaling']
alpha = [0.0001, 0.0002]


# Τυπώνουμε τις μετρικές του MLP από το Baseline Classification.

# In[ ]:


print("Accuracy: ", accuracy_score(y_test, mlp_pred))
print("\nMetrics: \n", classification_report(y_test, mlp_pred))


# Το dataset είναι πολύ μεγάλο, οπότε θα διαλέξουμε λιγότερα δείγματα.

# In[ ]:


#X_train, X_test, y_train, y_test = train_test_split(np_features, np_labels, test_size = 0.3)

from sklearn.utils import shuffle
sdata, starget = shuffle(np_features, np_labels, random_state=341976)
samples = 4300
data = sdata[0:samples-1,:]
target = starget[0:samples-1]

from sklearn.model_selection import train_test_split
# save previous train/test sets
#X_train_old, X_test_old, y_train_old, y_test_old = X_train, X_test, y_train, y_test
# split our sampled dataset
X_train2, X_test2, y_train2, y_test2 = train_test_split(data, target, test_size=0.3, random_state=20176)

imp = Imputer(strategy = 'mean', axis = 0)
imp = imp.fit(X_train)
X_train = imp.transform(X_train)
X_test = imp.transform(X_test)


# ***f1-micro***

# Θα δοκιμάσουμε όλους τους δυνατούς συνδυασμούς για τους μετασχηματιστές VarianceThreshold - StandrardScaler - ROS - PCA.

# Θα χρησιμοποιήσουμε τον παρακάτω κώδικα, προσθαφαιρώντας κάθε φορά μετασχηματιστές (μαζί με τις αντίστοιχες παραμέτρους τους).

# In[ ]:


#pipe_mlp = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('mlp', mlp)], memory = 'tmp')
#mlp_estimator1 = GridSearchCV(pipe_mlp, dict(selector__threshold=vthreshold_, pca__n_components=n_components_, mlp__hidden_layer_sizes=hidden_layer_sizes, mlp__activation=activation, mlp__solver=solver, mlp__max_iter=max_iter, mlp__learning_rate=learning_rate, mlp__alpha=alpha), cv=5, scoring='f1_micro', n_jobs=-1)
#mlp_estimator1.fit(X_train2, y_train2)
#mlp_est1_pred = mlp_estimator1.predict(X_test2)
#print("Accuracy: ", accuracy_score(y_test2, mlp_est1_pred))
#print("\nMetrics: \n", classification_report(y_test2, mlp_est1_pred))
#print("Optimum hyperparameters: ", mlp_estimator1.best_params_)


# Παίρνουμε τα κάτωθι αποτελέσματα:

# ('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('mlp', mlp)<br>f1-micro = 0.94<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'tanh', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'invscaling', 'mlp__max_iter': 50, 'mlp__solver': 'sgd', 'pca__n_components': 3, 'selector__threshold': 1000}

# ('scaler', scaler), ('sampler', ros), ('pca', pca), ('mlp', mlp)<br>f1-micro = 0.93<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'tanh', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (100,), 'mlp__learning_rate': 'invscaling', 'mlp__max_iter': 50, 'mlp__solver': 'sgd', 'pca__n_components': 3}

# ('selector', selector), ('sampler', ros), ('pca', pca), ('mlp', mlp)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0001, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 50, 'mlp__solver': 'sgd', 'pca__n_components': 3, 'selector__threshold': 5}

# ('selector', selector), ('scaler', scaler), ('pca', pca), ('mlp', mlp)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0001, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 50, 'mlp__solver': 'adam', 'pca__n_components': 3, 'selector__threshold': 5}

# ('selector', selector), ('scaler', scaler), ('sampler', ros), ('mlp', mlp)<br>f1-micro = 0.90<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (100,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 200, 'mlp__solver': 'adam', 'selector__threshold': 5}

# ('sampler', ros), ('pca', pca), ('mlp', mlp)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0001, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 50, 'mlp__solver': 'sgd', 'pca__n_components': 3}

# ('scaler', scaler), ('pca', pca), ('mlp', mlp)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0001, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 50, 'mlp__solver': 'sgd', 'pca__n_components': 3}

# ('scaler', scaler), ('sampler', ros), ('mlp', mlp)<br>f1-micro = 0.92<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0001, 'mlp__hidden_layer_sizes': (100,), 'mlp__learning_rate': 'invscaling', 'mlp__max_iter': 200, 'mlp__solver': 'adam'}

# ('selector', selector), ('pca', pca), ('mlp', mlp)<br>f1-micro = 0.69<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0001, 'mlp__hidden_layer_sizes': (100,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 200, 'mlp__solver': 'sgd', 'pca__n_components': 15, 'selector__threshold': 1000}

# ('selector', selector), ('sampler', ros), ('mlp', mlp)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 50, 'mlp__solver': 'sgd', 'selector__threshold': 1000}
# 

# ('selector', selector), ('scaler', scaler), ('mlp', mlp)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0001, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 50, 'mlp__solver': 'sgd', 'selector__threshold': 5}

# ('selector', selector), ('mlp', mlp)<br>f1-micro = 0.07<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0001, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 200, 'mlp__solver': 'sgd', 'selector__threshold': 1000}

# ('scaler', scaler), ('mlp', mlp)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0001, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 50, 'mlp__solver': 'sgd'}

# ('sampler', ros), ('mlp', mlp)<br>f1-micro = 0.86<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (100,), 'mlp__learning_rate': 'invscaling', 'mlp__max_iter': 200, 'mlp__solver': 'adam'}

# ('pca', pca), ('mlp', mlp)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 50, 'mlp__solver': 'sgd', 'pca__n_components': 3}

# ***('mlp', mlp)<br>f1-micro = 0.95<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0001, 'mlp__hidden_layer_sizes': (100,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 50, 'mlp__solver': 'sgd'}***

# Ορίζουμε λοιπόν τον τελικό estimator, και τυπώνουμε τις f1 μετρικές του και το accuracy.

# In[ ]:


mlp1 = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(100,), learning_rate='constant', max_iter=50, solver='sgd')

Mlp_mic = Pipeline(steps=[('mlp',mlp1)])

time1 = time.time()
Mlp_mic.fit(X_train, y_train)
time2 = time.time()
Mlp_mic_pred = Mlp_mic.predict(X_test)
time3 = time.time()

_,_,Mlp_f1_mic,_ = precision_recall_fscore_support(y_test, Mlp_mic_pred, average='micro')
_,_,f1_mac,_ = precision_recall_fscore_support(y_test, Mlp_mic_pred, average='macro')

print("f1-micro: ", Mlp_f1_mic)
print("f1-macro: ", f1_mac)
print("Accuracy: ", accuracy_score(y_test, Mlp_mic_pred))



# Και έχουμε τον εξής Confusion Matrix:

# In[ ]:


print_cm(Knn_mic_pred)


# Χρόνοι εκτέλεσης

# In[ ]:


train_time = time2 - time1
test_time = time3 - time2
print("train time: ", train_time)
print("test time: ", test_time)


# ***f1-macro***

# Θα δοκιμάσουμε όλους τους δυνατούς συνδυασμούς για τους μετασχηματιστές VarianceThreshold - StandrardScaler - ROS - PCA.

# Θα χρησιμοποιήσουμε τον παρακάτω κώδικα, προσθαφαιρώντας κάθε φορά μετασχηματιστές (μαζί με τις αντίστοιχες παραμέτρους τους).

# In[ ]:


#pipe_mlp = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('mlp', mlp)], memory = 'tmp')
#mlp_estimator2 = GridSearchCV(pipe_mlp, dict(selector__threshold=vthreshold_, pca__n_components=n_components_, mlp__hidden_layer_sizes=hidden_layer_sizes, mlp__activation=activation, mlp__solver=solver, mlp__max_iter=max_iter, mlp__learning_rate=learning_rate, mlp__alpha=alpha), cv=5, scoring='f1_macro', n_jobs=-1)
#mlp_estimator2.fit(X_train2, y_train2)
#mlp_est2_pred = mlp_estimator2.predict(X_test2)
#print("Accuracy: ", accuracy_score(y_test2, mlp_est2_pred))
#print("\nMetrics: \n", classification_report(y_test2, mlp_est2_pred))
#print("Optimum hyperparameters: ", mlp_estimator2.best_params_)


# Παίρνουμε τα κάτωθι αποτελέσματα:

# ('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('mlp', mlp)<br>f1-macro = 0.55<br>Βέλτιστοι υπερπαράμετροι: {'mlp__activation': 'relu', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (100,), 'pca__n_components': 15, 'mlp__solver': 'adam', 'mlp__learning_rate': 'constant', 'selector__threshold': 1000, 'mlp__max_iter': 200}

# ('scaler', scaler), ('sampler', ros), ('pca', pca), ('mlp', mlp)<br>f1-macro = 0.51<br>Βέλτιστοι υπερπαράμετροι: {'mlp__activation': 'relu', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (100,), 'pca__n_components': 15, 'mlp__solver': 'adam', 'mlp__learning_rate': 'constant', 'mlp__max_iter': 200}

# ('selector', selector), ('sampler', ros), ('pca', pca), ('mlp', mlp)<br>f1-macro = 0.56<br>Βέλτιστοι υπερπαράμετροι: {'mlp__activation': 'tanh', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (100,), 'pca__n_components': 15, 'mlp__solver': 'adam', 'mlp__learning_rate': 'invscaling', 'selector__threshold': 5, 'mlp__max_iter': 200}

# ('selector', selector), ('scaler', scaler), ('pca', pca), ('mlp', mlp)<br>f1-macro = 0.50<br>Βέλτιστοι υπερπαράμετροι: {'mlp__activation': 'tanh', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (50,), 'pca__n_components': 15, 'mlp__solver': 'sgd', 'mlp__learning_rate': 'invscaling', 'selector__threshold': 5, 'mlp__max_iter': 200}

# ('selector', selector), ('scaler', scaler), ('sampler', ros), ('mlp', mlp)<br>f1-macro = 0.63<br>Βέλτιστοι υπερπαράμετροι: {'mlp__activation': 'relu', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (100,), 'mlp__solver': 'adam', 'mlp__learning_rate': 'constant', 'selector__threshold': 5, 'mlp__max_iter': 200}

# ('sampler', ros), ('pca', pca), ('mlp', mlp)<br>f1-macro = 0.53<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'tanh', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (100,), 'mlp__learning_rate': 'invscaling', 'mlp__max_iter': 50, 'mlp__solver': 'adam', 'pca__n_components': 15}

# ('scaler', scaler), ('pca', pca), ('mlp', mlp)<br>f1-macro = 0.50<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'tanh', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'invscaling', 'mlp__max_iter': 50, 'mlp__solver': 'sgd', 'pca__n_components': 3}

# ***('scaler', scaler), ('sampler', ros), ('mlp', mlp)<br>f1-macro = 0.64<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0001, 'mlp__hidden_layer_sizes': (100,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 200, 'mlp__solver': 'adam'}***

# ('selector', selector), ('pca', pca), ('mlp', mlp)<br>f1-macro = 0.49<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (100,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 50, 'mlp__solver': 'adam', 'pca__n_components': 3, 'selector__threshold': 5}

# ('selector', selector), ('sampler', ros), ('mlp', mlp)<br>f1-macro = 0.56<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0001, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'invscaling', 'mlp__max_iter': 50, 'mlp__solver': 'adam', 'selector__threshold': 5}

# ('selector', selector), ('scaler', scaler), ('mlp', mlp)<br>f1-macro = 0.48<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'tanh', 'mlp__alpha': 0.0001, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'invscaling', 'mlp__max_iter': 200, 'mlp__solver': 'adam', 'selector__threshold': 5}

# ('selector', selector), ('mlp', mlp)<br>f1-macro = 0.49<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0001, 'mlp__hidden_layer_sizes': (100,), 'mlp__learning_rate': 'invscaling', 'mlp__max_iter': 200, 'mlp__solver': 'adam', 'selector__threshold': 5}

# ('scaler', scaler), ('mlp', mlp)<br>f1-macro = 0.50<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'tanh', 'mlp__alpha': 0.0001, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'invscaling', 'mlp__max_iter': 50, 'mlp__solver': 'adam'}

# ('sampler', ros), ('mlp', mlp)<br>f1-macro = 0.61<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'invscaling', 'mlp__max_iter': 200, 'mlp__solver': 'adam'}

# ('pca', pca), ('mlp', mlp)<br>f1-macro = 0.48<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (100,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 50, 'mlp__solver': 'adam', 'pca__n_components': 3}

# ('mlp', mlp)<br>f1-macro = 0.54<br>Βέλτιστοι υπερπαράμετροι:  {'mlp__activation': 'relu', 'mlp__alpha': 0.0002, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 200, 'mlp__solver': 'adam'}

# Ορίζουμε λοιπόν τον τελικό estimator, και τυπώνουμε τις f1 μετρικές του και το accuracy.

# In[ ]:


scaler1 = StandardScaler()
ros1 = RandomOverSampler()
mlp1 = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(100,), learning_rate='constant', max_iter=200, solver='adam')

Mlp_mac = Pipeline(steps=[('scaler', scaler1), ('sampler', ros1), ('mlp',mlp1)])

time1 = time.time()
Mlp_mac.fit(X_train, y_train)
time2 = time.time()
Mlp_mac_pred = Mlp_mac.predict(X_test)
time3 = time.time()

_,_,f1_mic,_ = precision_recall_fscore_support(y_test, Mlp_mac_pred, average='micro')
_,_,Mlp_f1_mac,_ = precision_recall_fscore_support(y_test, Mlp_mac_pred, average='macro')

print("f1-micro: ", f1_mic)
print("f1-macro: ", Mlp_f1_mac)
print("Accuracy: ", accuracy_score(y_test, Mlp_mac_pred))



# Και έχουμε τον εξής Covariance Matrix:

# In[ ]:


print_cm(Knn_mic_pred)


# Χρόνοι εκτέλεσης

# In[ ]:


train_time = time2 - time1
test_time = time3 - time2
print("train time: ", train_time)
print("test time: ", test_time)


# ##3.

# Εκτύπωση plots f1 μετρικές

# Για f1-micro:

# In[ ]:


objects = ('GNB', 'kNN', 'MLP')
y_pos = np.arange(len(objects))

perf = [Gnb_f1_mic, Knn_f1_mic, Mlp_f1_mic]
bar_plot_function('Average Micro F1-Score', perf)


# Για f1-macro:

# In[ ]:


perf = [Gnb_f1_mac, Knn_f1_mac, Mlp_f1_mac]
bar_plot_function('Average Macro F1-Score', perf)


# ##4.

# Μεταβολή επίδοσης των ταξινομητών

# *Gaussian Naive Bayes*

# In[ ]:


print("Gaussian Naive Bayes\n")
print("         ", "f1-micro", "          ", "f1-macro")
print("before : ", gnb_f1_mic, gnb_f1_mac)
print("after  : ", Gnb_f1_mic, "", Gnb_f1_mac)


# *kNN*

# In[ ]:


print("kNN\n")
print("         ", "f1-micro", "         ", "f1-macro")
print("before : ", knn_f1_mic, knn_f1_mac)
print("after  : ", Knn_f1_mic, Knn_f1_mac)


# *Multi-Layer Perceptron*

# In[ ]:


print("Multi-Layer Perceptron\n")
print("         ", "f1-micro", "         ", "f1-macro")
print("before : ", mlp_f1_mic, mlp_f1_mac)
print("after  : ", Mlp_f1_mic, Mlp_f1_mac)


# ##5.

# Παρατηρούμε ότι η μετρική f1-micro φτάνει σε πολύ υψηλό ποσοστό και στους 3 ταξινομητές. Αυτό συμβαίνει διότι είναι πολύ εύκολο να προβλεπτεί η κλάση 0 λόγω της μεγάλης ανισορροπίας που υπάρχει (95-5).<br>
# Για αυτό το λόγο όμως είναι αντίστοιχα δύσκολο να προβλεπτεί η κλάση 1, και για αυτό έχοντας χαμηλά ποσοστά σε αυτήν, βλέπουμε ότι το f1-macro πιάνει αρκετά χαμηλά ποσοστά και στους 3 ταξινομητές (ειδικά στον Gaussian). Αυτό φαίνεται εύκολα και από τα μητρώα Confusion, όπου έχουμε μεγάλο αριθμό στη θέση (2,1) σε σχέση με την (2,2). Δηλαδή, πολλά δείγματα της κλάσης 2 δεν προβλεύθηκαν σωστά.<br>
# Τις μεγαλύτερες μεταβολές σε απόδοση τις είχε εμφανώς ο GNB, καθότι πριν την βελτιστοποίησή του δεν έπιανε ούτε 10% σε f1 μετρικές.<br>
# Όσον αφορά τους χρόνους εκτέλεσης, παρατηρούμε μικρούς χρόνους για τον GNB τόσο στο fit όσο και στο predict. Στον kNN έχουμε μικρούς χρόνους στο fit, αλλά μεγάλους στο predict. Και στον MLP έχουμε ακριβώς το αντίστροφο. Επίσης παρατηρούμε ότι ανάλογα με το πόσο μεγάλο είναι το pipeline (αριθμός transformers) αυξάνονται και οι χρόνοι εκτέλεσης.

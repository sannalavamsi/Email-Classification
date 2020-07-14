#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imblearn
from imblearn import over_sampling

from imblearn.over_sampling import SMOTE


# In[2]:


email = pd.read_csv('/Users/apple/Downloads/DS_Projects/Email Project/emails1(1)')


# In[3]:


email.head()


# # EDA

# In[4]:


email.dtypes


# In[5]:


email1=email.drop(email.columns[[0,1,2]],axis=1)
email1.head()


# In[6]:


email1.info()


# In[7]:


# Finding numner of null values
email1.isnull().sum()


# In[8]:


#Shape of email 
email1.shape


# In[9]:


#Total numnber of unique in each Variable
email1.nunique()


# In[10]:


#Drop duplicate rows in Pandas based on column
df=email1.drop_duplicates(subset='content', keep="first")
df.shape


# In[12]:


#count of Class Variable 
df["Class"].value_counts()


# In[13]:


#finding the email["Class"] value_counts in percentage
df["Class"].value_counts()*100/len(df)


# In[14]:


#Histogram of email["Class"] to find whether data is balanced
df["Class"].hist(figsize=(15,5))
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show


# In[15]:


from textblob import TextBlob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity


# In[ ]:


df['polarity'] = df['content'].apply(pol)
df['subjectivity'] = df['content'].apply(sub)
df.head()


# In[18]:


ax=sns.distplot(df['polarity'],bins=50)
ax.set(xlabel="polarity", ylabel = "probability")


# In[19]:


#text length
df['text_len'] = df['content'].str.len()   ## this also includes spaces
df.head()


# In[20]:


df['text_len'].describe()


# In[21]:


#Histogram of df['text_len']
df['text_len'].plot(kind='hist',bins=200,title='text_length Distribution')


# In[22]:


ax=sns.distplot(df['text_len'],bins=100)
ax.set(xlabel="count", ylabel = "probability",title='text_length Distribution')


# In[23]:


# Number of Words
df['word_count'] = df['content'].apply(lambda x: len(str(x).split(" ")))
df.loc[:,('content','word_count')].head()


# In[24]:


df['word_count'].describe()


# In[26]:


#Histogram of email["Class"] to find whether data is balanced
df["word_count"].plot(kind='hist',bins=200,title='Words count in content Distribution')


# In[27]:


ax1=sns.distplot(df['word_count'],bins=100)
ax1.set(xlabel="count", ylabel = "probability",title='Word count in content distribution')


# In[28]:


df.head()


# In[29]:


df.isnull().sum()


# # Pre-processing
# 

# In[11]:


import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
wnl = WordNetLemmatizer()
from textblob import TextBlob
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[12]:


corpus = []
def data_clean(data):
    wnl = nltk.WordNetLemmatizer()
    for i in data:
        review = re.sub('[^a-zA-Z]',' ',str(i))#remove all except a-z A-Z text
        review = review.lower()#to lowercase
        review = review.split()#split the text
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = [wnl.lemmatize(word) for word in review]
        review = ' '.join(review)
        corpus.append(review)


# In[13]:


data_clean(df['content'])


# In[14]:


corpus


# In[51]:


df['cleaned_content']=corpus


# In[52]:


df.head()


# In[53]:


print('5 random reviews with the highest positive sentiment polarity: \n')
cl = df.loc[df.polarity == 1, ['cleaned_content']].sample(5).values
for c in cl:
    print(c[0])


# In[55]:


print('5 random reviews with the most neutral sentiment(zero) polarity: \n')
cl = df.loc[df.polarity == 0, ['cleaned_content']].sample(5).values
for c in cl:
    print(c[0])


# In[57]:


print(' reviews with the most negative polarity: \n')
cl = df.loc[df.polarity == -1, ['cleaned_content']].sample(2).values
for c in cl:
    print(c[0])


# In[34]:


len(corpus)


# In[ ]:





# In[25]:


# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X = cv.fit_transform(corpus)
X.shape


# In[41]:


import pickle
# Saving model to disk
pickle.dump(cv, open('transform1.pkl','wb'))


# In[26]:


#Y= independent variable
y=df.iloc[:,1:2]
Y=np.ravel(y)
Y


# In[27]:


Y.shape


# In[28]:


#train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)


# In[29]:


print(X_train.shape)
print(Y_train.shape)


# # MultinomialNB without using oversampling

# In[30]:


from sklearn.naive_bayes import MultinomialNB
MNB =MultinomialNB()
MNB.fit(X_train,Y_train)


# In[31]:


y_predict=MNB.predict(X_test)
y_predict


# In[32]:


from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
print(confusion_matrix(Y_test,y_predict))
print(classification_report(Y_test,y_predict))
print(accuracy_score(Y_test,y_predict))


# # MultinomialNB using oversampling

# In[33]:


abusive=df[df['Class']=='Abusive']
non_abusive=df[df['Class']=='Non Abusive']
print(abusive.shape,non_abusive.shape)


# In[34]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=4)
X_train1, y_train1 = sm.fit_sample(X_train, Y_train)


# In[35]:


X_train1.shape,y_train1.shape


# In[36]:



from collections import Counter
print('Original dataset shape {}'.format(Counter(Y_train)))
print('Resampled dataset shape {}'.format(Counter( y_train1)))


# In[37]:


MNB_sm =MultinomialNB()
MNB_sm.fit(X_train1, y_train1)


# In[38]:


y_predict_sm=MNB_sm.predict(X_test)
y_predict_sm


# In[39]:


print(confusion_matrix(Y_test,y_predict_sm))
print(classification_report(Y_test,y_predict_sm))
print(accuracy_score(Y_test,y_predict_sm))


# In[40]:


from sklearn.model_selection import cross_val_score
MNB_sm_scores = cross_val_score(MNB_sm,X_train1, y_train1, cv=5)
print(MNB_sm_scores)
print(np.average(MNB_sm_scores))


# In[42]:


import pickle
# Saving model to disk
pickle.dump(MNB_sm, open('MNB_sm_model1.pkl','wb'))


# In[ ]:





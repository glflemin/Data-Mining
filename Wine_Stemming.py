
# coding: utf-8

# In[3]:


import pandas as pd
import nltk
import numpy
import re
import string

wine_data = pd.read_csv("winemag-data-130k-v2.csv")
wine_data.head()


# In[7]:

# Create initial documents list
doc = wine_data.description

# Remove punctuation, then tokenize documents
punc = re.compile( '[%s]' % re.escape( string.punctuation ) )
term_vec = [ ]

for d in doc:
    d = d.lower()
    d = punc.sub( '', d )
    term_vec.append( nltk.word_tokenize( d ) )

# Print resulting term vectors
#for vec in term_vec:
#    print(vec)


# In[8]:

# Remove stop words from term vectors
stop_words = nltk.corpus.stopwords.words( 'english' )

for i in range( 0, len( term_vec ) ):
    term_list = [ ]

    for term in term_vec[ i ]:
        if term not in stop_words:
            term_list.append( term )

    term_vec[ i ] = term_list

# Print term vectors with stop words removed
#for vec in term_vec:
#    print(vec)


# In[9]:

# Porter stem remaining terms
porter = nltk.stem.porter.PorterStemmer()

for i in range( 0, len( term_vec ) ):
    for j in range( 0, len( term_vec[ i ] ) ):
        term_vec[ i ][ j ] = porter.stem( term_vec[ i ][ j ] )


# In[10]:


print(term_vec[0])


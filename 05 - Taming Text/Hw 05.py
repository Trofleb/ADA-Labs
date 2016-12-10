
# coding: utf-8

# ## Some useful imports

# In[ ]:

import nltk
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Getting email text

# In[ ]:

emails = pd.read_csv("./hillary-clinton-emails/Emails.csv")


# In[ ]:

emails.head()


# Concatenating all the e-mails into one big text.

# In[ ]:

original_text = "".join(emails.RawText)


# ### First raw word count 

# In[ ]:

def show_wordCloud(text):
    # Generate a word cloud image
    wordcloud = WordCloud().generate(text)

    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

show_wordCloud(original_text)


# ## Processing pipeline

# In[ ]:

tokens = np.array(nltk.word_tokenize(original_text))


# In[ ]:

def lower_case(tokens):
    return np.array([token.lower() for token in tokens])

tokens = lower_case(tokens)


# In[ ]:

def remove_stop_words(tokens):
    from nltk.corpus import stopwords
    return tokens[np.in1d(ar1=tokens, ar2=stopwords.words("english"), invert=True)]

tokens = remove_stop_words(tokens)


# In[ ]:

from nltk.stem import WordNetLemmatizer
def lemmatization(tokens):
    wordnet_lemmatizer = WordNetLemmatizer()
    return np.array([wordnet_lemmatizer.lemmatize(token) for token in tokens])


# In[ ]:

from nltk.stem.porter import PorterStemmer

def stemming(tokens):
    stemmer = PorterStemmer()
    return np.array([stemmer.stem(token) for token in tokens])

tokens = stemming(tokens)


# In[ ]:

text = " ".join(tokens)


# In[ ]:

show_wordCloud(text)


# In[ ]:




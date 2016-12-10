
# coding: utf-8

# In[10]:

import nltk
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:

emails = pd.read_csv("./hillary-clinton-emails/Emails.csv")


# In[6]:

emails.head()


# In[9]:

text = "".join(emails.RawText)


# In[12]:

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[ ]:




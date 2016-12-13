
# coding: utf-8

# ## Some useful imports

# In[158]:

import nltk
import re
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel


# ## Getting email text

# In[159]:

emails = pd.read_csv("./hillary-clinton-emails/Emails.csv", usecols=["RawText"])


# In[160]:

emails.head()


# Concatenating all the e-mails into one big text.

# In[161]:

texts = [text for text in emails.RawText]
full_text = " ".join(texts)


# ### First raw word count 

# In[162]:

def show_wordCloud(text):
    # lower max_font_size
    wordcloud = WordCloud(
        max_font_size=60, 
        background_color='white',
        width=800,
        height=600
    ).generate(text)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

show_wordCloud(full_text)


# We can see different kind of patterns:
# - generic words in emails: re, part, date, sent, subject, message
# - classical words: will, want, many
# - uppercase vs lowercase
# - accronyms
# - etc.

# Pros:
# - no processing, pure data: no risk of having drop important part or change the meaning
# 
# Cons:
# - difficult to extract important words: they are hidden in the mass
# - not homogenous: year and years are counted separately
# - noisy: some words do not have a single meaning themselves, the need a context (e.g. stop words)

# ## Processing pipeline

# Here is the pipeline we choosed to implement:
# - tokenization
# - lowercasing
# - stop word removal
# - lemmatization
# - stemming
# - url/email removal: although they not appear in the world cloud url are often present in mails and can be blacklisted with `./_@:` which are usally part of desired words

# In[166]:

def lower_case(texts_tokenized):
    new_tokens = []
    for tokens in texts_tokenized:
        new_tokens.append([token.lower() for token in tokens])
    return new_tokens


# In[168]:

def remove_stop_words(texts_tokenized):
    new_tokens = []
    for tokens in texts_tokenized:
        new_tokens.append(np.array(tokens)[np.in1d(tokens, stopwords.words("english"), invert=True)])
    return new_tokens


# In[170]:

def lemmatization(texts_tokenized):
    new_tokens = []
    wordnet_lemmatizer = WordNetLemmatizer()
    for tokens in texts_tokenized:
        new_tokens.append([wordnet_lemmatizer.lemmatize(token) for token in tokens])
    return new_tokens


# In[172]:

def stemming(texts_tokenized):
    new_tokens = []
    stemmer = PorterStemmer()
    for tokens in texts_tokenized:
        new_tokens.append([stemmer.stem(token) for token in tokens])
    return new_tokens


# In[174]:

def remove_special_chars(texts_tokenized):
    blacklist_regex = r'\W' # Matches every token that is a non-alphanumeric character
    new_tokens = []
    for tokens in texts_tokenized:
        new_tokens.append([token for token in tokens if not re.match(blacklist_regex, token)])
    return new_tokens


# In[176]:

def remove_email_terminology(texts_tokenized):
    blacklist_terms = ['subject', 'cc', 'cci', 'sent', 'date', 're', 'mailto']
    new_tokens = []
    for tokens in texts_tokenized:
        new_tokens.append([token for token in tokens if not np.in1d(list(token), blacklist_terms).any()])
    return new_tokens


# In[180]:

texts_tokenized = [nltk.word_tokenize(text) for text in texts]
texts_tokenized = lower_case(texts_tokenized)
texts_tokenized = remove_stop_words(texts_tokenized)
texts_tokenized = lemmatization(texts_tokenized)
texts_tokenized = stemming(texts_tokenized)
texts_tokenized = remove_special_chars(texts_tokenized)
texts_tokenized = remove_email_terminology(texts_tokenized)


# In[181]:

texts2 = [" ".join(tokens) for tokens in texts_tokenized]
full_text2 = " ".join(texts2)


# In[182]:

show_wordCloud(full_text)


# The result has improved a lot, however there are still issues:
# - generic words in emails: this would require extending the stop words list with targetted words and would require more analysis
# - classical words: this would require extending the stop words list or use "zpif score" to avoid considering them
# - accronyms: this would require to generate a list corresponding term which is indeed a hard problem

# Pros:
# - more variaty, more homogenous
# - similar words put together whenever possible
# - avoid stop words
#     
# Cons:
# - might have influence the content: in case pipeline/methods introducing some biais
# - still some issues with normalization: n't should be here

# ## Only nouns

# One could also try to keep only (proper) nouns as verbs, adjectives and adverbs might have a very generic meaning (e.g. also, send). Using a POS tagging just after tokenization, we can discard the others.

# In[184]:

def keep_nouns(texts_tokenized):
    to_keep_tag = ['NN', 'NNS', 'NNP', 'NNPS']
    new_tokens = []
    for tokens in texts_tokenized:
        tags = nltk.pos_tag(set(tokens))
        to_keep_tokens = [tok for tok, tag in tags if tag in to_keep_tag]
        new_tokens.append(np.array(tokens)[np.in1d(tokens, to_keep_tokens)])
    return new_tokens


# In[185]:

texts_tokenized2 = [ nltk.word_tokenize(text) for text in texts]
texts_tokenized2 = keep_nouns(texts_tokenized2)
texts_tokenized2 = lower_case(texts_tokenized2)
texts_tokenized2 = remove_stop_words(texts_tokenized2)
texts_tokenized2 = lemmatization(texts_tokenized2)


# In[186]:

# Changing variable name so that we can use texts_tokenized2 without stemming in Part 3
texts_tokenized2_clean = stemming(texts_tokenized2)
texts_tokenized2_clean = remove_special_chars(texts_tokenized2_clean)
texts_tokenized2_clean = remove_email_terminology(texts_tokenized2_clean)


# In[193]:

texts3 = [" ".join(tokens) for tokens in texts_tokenized2_clean]
full_text3 = " ".join(texts3)


# In[194]:

show_wordCloud(full_text3)


# Although this is more aggressive than the previous trials, this world cloud gives a better overview of the general content. The non-nouns are might be due to english verbs/nouns duality.

# # 3. Topic modelling

# ## Cleaning

# To get meaningful topics, we remove the words that occur in almost every e-mail and that do not convey information about the topic discussed in it (e.g words from the banner present in many of them: "unclassified u.s department of state...")

# In[196]:

def remove_unwanted_tokens(tokens):
    # Regular expression matching all single non-alphanumeric characters (\W), 
    # case numbers in f-xxxx-yyyy (f-\d{4}-\d*) format, tokens such as "c1234.." (\w\d+), 
    # tokens with less than 3 letters (^\w{1,2}$), e-mail addresses (\w+\.com) and all other listed tokens
    reoccuring_tokens = ["case","doc","state", "http","foia","waiver","agreement","sensitive","information",                         "unclassified","date","subject","cc","cci","sent","re","mailto","department"]
    reoccuring_tokens_re = '|'.join(reoccuring_tokens)
    blacklist_re = re.compile(r"("+reoccuring_tokens_re+"|\W|u\.s|f-\d{4}-\d*|\w\d+|^\w{1,2}$)|\w+\.com" )
    
    result = []
    for t in tokens:
        result.append([token for token in t if not re.match(blacklist_re, token)])
    
    return result


# We skipped the stemming step in order to keep a big enough part of the words. That way we will be able to understand their actual meaning and origin.

# In[197]:

texts_tokenized3 = remove_unwanted_tokens(texts_tokenized2)


# Here is a comparison before and after removing unwanted tokens. <br />
# It seems like we lose many words, but it is actually only a few unique words appearing many times as well as some tokens smaller than 3 characters and the non-alphanumeric ones. This is fine as they do not add meaning to the e-mail. <br />
# The dictionnary we build later on has approximately 2000 tokens less after applying that function than before (from ~37500 tokens to ~35500 tokens)

# In[198]:

' '.join(texts_tokenized2[0])


# In[199]:

' '.join(texts_tokenized3[0])


# ## Corpus

# We create the dictionnary from the processed e-mail texts using gensim.corpora.dictionary.Dictionary 

# In[200]:

corpus_dictionary = Dictionary(texts_tokenized3)
print(corpus_dictionary)


# With that dictionnary we create a corpus made of the bag-of-words representation of each text

# In[201]:

corpus = [corpus_dictionary.doc2bow(tokens) for tokens in texts_tokenized3]


# ## Model

# Next, we train an Lda model on that corpus using gensim.models.LdaModel and the previously built dictionary. <br />
# We set the hyperparameters alpha and eta to 'auto' so that gensim manages and adapts them to our data. We also do 5 passes on the corpus to improve the correctness/accuracy of the topics.

# In[204]:

#model = LdaModel(corpus, id2word=corpus_dictionary, num_topics=42, passes=5, alpha='auto', eta='auto', minimum_probability=0.02)


# In[ ]:

model = LdaModel.load('best_model/best_model')


# In[ ]:

model.save('best_model/best_model')


# In[206]:

model.show_topics(num_topics=20, num_words=10, formatted=False)


# After many tries, we find that the parameters giving us the most meaningful topics at first sight are:
#  - num_topics: 42
#  - passes: 5
#  - alpha: 'auto'
#  - eta: 'auto'
#  - minimum_probability: 0.02
#  
# The many models tested are not available because they took way too much space to upload (folder "tests_models")... <br />
# ![Image](folder_size.png)

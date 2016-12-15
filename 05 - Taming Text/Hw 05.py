
# coding: utf-8

# ## Some useful imports

# In[3]:

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

# In[5]:

emails = pd.read_csv("./hillary-clinton-emails/Emails.csv", usecols=["RawText"])


# In[3]:

emails.head()


# Concatenating all the e-mails into one big text.

# In[6]:

texts = [text for text in emails.RawText]
full_text = " ".join(texts)


# ### First raw word count 

# In[7]:

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

# In[8]:

def lower_case(texts_tokenized):
    new_tokens = []
    for tokens in texts_tokenized:
        new_tokens.append([token.lower() for token in tokens])
    return new_tokens


# In[9]:

def remove_stop_words(texts_tokenized):
    new_tokens = []
    for tokens in texts_tokenized:
        new_tokens.append(np.array(tokens)[np.in1d(tokens, stopwords.words("english"), invert=True)])
    return new_tokens


# In[10]:

def lemmatization(texts_tokenized):
    new_tokens = []
    wordnet_lemmatizer = WordNetLemmatizer()
    for tokens in texts_tokenized:
        new_tokens.append([wordnet_lemmatizer.lemmatize(token) for token in tokens])
    return new_tokens


# In[11]:

def stemming(texts_tokenized):
    new_tokens = []
    stemmer = PorterStemmer()
    for tokens in texts_tokenized:
        new_tokens.append([stemmer.stem(token) for token in tokens])
    return new_tokens


# In[12]:

def remove_special_chars(texts_tokenized):
    blacklist_regex = r'\W' # Matches every token that is a non-alphanumeric character
    new_tokens = []
    for tokens in texts_tokenized:
        new_tokens.append([token for token in tokens if not re.match(blacklist_regex, token)])
    return new_tokens


# In[13]:

def remove_email_terminology(texts_tokenized):
    blacklist_terms = ['subject', 'cc', 'cci', 'sent', 'date', 're', 'mailto']
    new_tokens = []
    for tokens in texts_tokenized:
        new_tokens.append([token for token in tokens if not np.in1d(list(token), blacklist_terms).any()])
    return new_tokens


# In[14]:

texts_tokenized = [nltk.word_tokenize(text) for text in texts]
texts_tokenized = lower_case(texts_tokenized)
texts_tokenized = remove_stop_words(texts_tokenized)
texts_tokenized = lemmatization(texts_tokenized)
texts_tokenized = stemming(texts_tokenized)
texts_tokenized = remove_special_chars(texts_tokenized)
texts_tokenized = remove_email_terminology(texts_tokenized)


# In[15]:

texts2 = [" ".join(tokens) for tokens in texts_tokenized]
full_text2 = " ".join(texts2)


# In[16]:

show_wordCloud(full_text2)


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

# In[17]:

def keep_nouns(texts_tokenized):
    to_keep_tag = ['NN', 'NNS', 'NNP', 'NNPS']
    new_tokens = []
    for tokens in texts_tokenized:
        tags = nltk.pos_tag(set(tokens))
        to_keep_tokens = [tok for tok, tag in tags if tag in to_keep_tag]
        new_tokens.append(np.array(tokens)[np.in1d(tokens, to_keep_tokens)])
    return new_tokens


# In[18]:

texts_tokenized2 = [ nltk.word_tokenize(text) for text in texts]
texts_tokenized2 = keep_nouns(texts_tokenized2)
texts_tokenized2 = lower_case(texts_tokenized2)
texts_tokenized2 = remove_stop_words(texts_tokenized2)
texts_tokenized2 = lemmatization(texts_tokenized2)


# In[19]:

# Changing variable name so that we can use texts_tokenized2 without stemming in Part 3
texts_tokenized2_clean = stemming(texts_tokenized2)
texts_tokenized2_clean = remove_special_chars(texts_tokenized2_clean)
texts_tokenized2_clean = remove_email_terminology(texts_tokenized2_clean)


# In[20]:

texts3 = [" ".join(tokens) for tokens in texts_tokenized2_clean]
full_text3 = " ".join(texts3)


# In[21]:

show_wordCloud(full_text3)


# Although this is more aggressive than the previous trials, this world cloud gives a better overview of the general content. The non-nouns are might be due to english verbs/nouns duality.

# We wanted here to achieve some sort of Luhn's analysis. We could also have taken word frequencies for each document and only kept the words which are not frequent in all mails.

# In[18]:

from IPython.display import Image
Image(filename="./luhn's analysis.png")


# ## Part 2

# In[78]:

import pycountry


# We are going to use the  sentiment analyser from Vader first. After looking at the results of the demo function we will be using the compound part of the result which gives the entensity of the sentiment and is positive or negative with respect to the type of sentiment (positive, negative)

# We confirmed our choice with this source.
# https://www.linkedin.com/pulse/sentiment-analysis-using-vader-muthuraj-kumaresan

# In[363]:

from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[364]:

sid = SentimentIntensityAnalyzer()


# We compute for each mail the sentiment analysis and store it in a csv file so that we don't have to redo it (very slow...)

# In[286]:

sentiment_df = pd.DataFrame(columns=["compound", "neutral", "negative", "positive"])

import progressbar

bar = progressbar.ProgressBar(max_value=len(emails.RawText))

i = 0
for email in emails.RawText:
    bar.update(i)
    score = sid.polarity_scores(email)
    sentiment_df.loc[i] = (score["compound"], score["neu"], score["neg"], score["pos"])
    i += 1


# In[287]:

sentiment_vader_df.to_csv("sentiment_vader.csv")


# We read it back from the csv.

# In[365]:

sentiment_vader_df = pd.read_csv("./sentiment_vader.csv", index_col=0)


# In[366]:

sentiment_vader_df.describe()


# We process the dataset for the contries so that we can use it easily afterwards.

# In[367]:

countries = [(country.name.lower(), country.alpha_2.lower(), country.alpha_3.lower()) for country in pycountry.countries]


# we initialize a dataframe for the sentiment per country

# In[368]:

sentiment_per_country = pd.DataFrame(columns=["country", "pos","neg"]).set_index("country")


# In[369]:

for name, _, _ in countries:
    sentiment_per_country.loc[name] = 0


# we populate it with the sentiment per mail that contains the country, you may notice that this is not the best way to do things as the country may not be related to the sentiment of the whole mail doing it by sentence would be more fair but it would be much more long to compute.

# In[381]:

import progressbar

bar = progressbar.ProgressBar(max_value=len(emails.RawText))

i = 0
for email in emails.RawText:
    bar.update(i)
    add = 0
    sentiment = sentiment_vader_df.loc[i]["compound"]
    
    tokens = nltk.word_tokenize(email)
    nouns = keep_nouns([tokens])[0]
    country_found = []
    # find countries
    for noun in nouns:
        lc_noun = noun.lower()
        index = 0
        for name, a2, a3 in countries:
            if name not in country_found and (lc_noun == name or lc_noun == a2 or lc_noun == a3):
                country_found.append(name)
                if sentiment >= 0:
                    sentiment_per_country.pos.loc[name] += 1
                if sentiment < 0:
                    sentiment_per_country.neg.loc[name] += 1
    i += 1


# In[382]:

sentiment_per_country.describe()


# At this point you may realise that we have huge values in the max for both pos and neg, we will see this in more details a bit later.

# Because we have a lot of countries and we want the graph to readable we will only show the most important countries

# In[383]:

quantile = 0.9


# In[384]:

(1 - quantile)* len(countries)


# In[385]:

sentiment_per_country_pos = sentiment_per_country[(sentiment_per_country.pos > sentiment_per_country.pos.quantile(quantile)) | (sentiment_per_country.neg > sentiment_per_country.neg.quantile(quantile))]


# In[386]:

sentiment_per_country_pos.shape


# In[387]:

sentiment_per_country_pos.plot.bar(y="pos")
sentiment_per_country_pos.plot.bar(y="neg", color="r")
plt.show()


# We see here a problem with our simple implementation. There are some countries who are much higher than they should be. To show why let's look at their names and alpha2 : 
# 
# Armenia -> am, 
# 
# Cocos (keeling) island -> CC, 
# 
# India -> in, 
# 
# Réunion -> re, 
# 
# Saint Pierre and Miquelon -> pm 
# 

# There are very common words in english that don't seem to be taken in account in the noun filter we have. We found out that it was because they were in caps.
# 
# We decided to remove those words completly as these countries (maybe not India) are not so relevant than the other mentioned for example the US.
# 
# Therefore if you search for the feeling of Hilary Clinton towards these countries specifically please discard this study.

# In[275]:

for name, _, _ in countries:
    sentiment_per_country.loc[name] = 0


# In[312]:

bar = progressbar.ProgressBar(max_value=len(emails.RawText))

i = 0
for email in emails.RawText:
    bar.update(i)
    add = 0
    sentiment = sentiment_vader_df.loc[i]["compound"]
    
    tokens = nltk.word_tokenize(email.lower())
    nouns = keep_nouns([tokens])[0]
    nouns = nouns[np.in1d(nouns, ["am","cc","re","in","pm"], invert=True)]
    country_found = []
    # find countries
    for noun in nouns:
        index = 0
        for name, a2, a3 in countries:
            if name not in country_found and (noun == name or noun == a2 or noun == a3):
                country_found.append(name)
                if sentiment >= 0:
                    sentiment_per_country.pos.loc[name] += 1
                if sentiment < 0:
                    sentiment_per_country.neg.loc[name] += 1
    i += 1


# In[356]:

sentiment_per_country.describe()


# In[360]:

sentiment_per_country_pos = sentiment_per_country[(sentiment_per_country.pos > sentiment_per_country.pos.quantile(quantile)) | (sentiment_per_country.neg > sentiment_per_country.neg.quantile(quantile))]


# In[361]:

sentiment_per_country_pos.shape


# In[362]:

sentiment_per_country_pos.plot.bar(y="pos")
sentiment_per_country_pos.plot.bar(y="neg", color="r")
plt.show()


# With this we have a much cleaner look at the frequencies of positive and negative emails of Clinton

# In[389]:

from nltk.corpus import opinion_lexicon
pos_lex = opinion_lexicon.positive()
neg_lex = opinion_lexicon.negative()


# In[230]:

sentiment_Liu_Hu_df = pd.DataFrame(columns=["sentiment", "count_pos", "count_neg"])
bar = progressbar.ProgressBar(max_value=len(emails.RawText))

i = 0
for email in emails.RawText:
    bar.update(i)
    tokens = nltk.word_tokenize(email)
    count_pos = np.sum(np.in1d(tokens, pos_lex))
    count_neg = np.sum(np.in1d(tokens, neg_lex))
    
    if count_pos > count_neg:
        sent = 1
    else:
        sent = -1
    
    sentiment_df.loc[i] = (sent, count_pos, count_neg)
    i += 1


# In[ ]:

sentiment_Liu_Hu_df.to_csv("sentiment_Liu_Hu.csv")


# We now do the same for the Liu Hu method which is "just" a count of positive and negative word in precomputed lexicon.

# In[390]:

sentiment_Liu_Hu_df = pd.read_csv("./sentiment_Liu_Hu.csv", index_col=0)


# In[391]:

sentiment_Liu_Hu_df.describe()


# In[392]:

countries = [(country.name.lower(), country.alpha_2.lower(), country.alpha_3.lower()) for country in pycountry.countries]


# In[393]:

sentiment_per_country = pd.DataFrame(columns=["country", "pos","neg"]).set_index("country")


# In[394]:

for name, _, _ in countries:
    sentiment_per_country.loc[name] = 0


# In[338]:

i = 0
bar = progressbar.ProgressBar(max_value=len(emails.RawText))

for email in emails.RawText:
    bar.update(i)
    add = 0
    sentiment = sentiment_Liu_Hu_df.loc[i]["sentiment"]
    
    tokens = nltk.word_tokenize(email.lower())
    nouns = keep_nouns([tokens])[0]
    nouns = nouns[np.in1d(nouns, ["am","cc","re","in","pm"], invert=True)]
    country_found = []
    # find countries
    for noun in nouns:
        lc_noun = noun.lower()
        index = 0
        for name, a2, a3 in countries:
            if name not in country_found and (lc_noun == name or lc_noun == a2 or lc_noun == a3):
                country_found.append(name)
                if sentiment >= 0:
                    sentiment_per_country.pos.loc[name] += 1
                if sentiment < 0:
                    sentiment_per_country.neg.loc[name] += 1
    i += 1


# In[339]:

sentiment_per_country.describe()


# In[351]:

sentiment_per_country_pos = sentiment_per_country[(sentiment_per_country.pos > sentiment_per_country.pos.quantile(quantile)) | (sentiment_per_country.neg > sentiment_per_country.neg.quantile(quantile))]


# In[352]:

sentiment_per_country_pos.shape


# In[353]:

sentiment_per_country_pos.plot.bar(y="pos")
sentiment_per_country_pos.plot.bar(y="neg", color="r")
plt.show()


# And here we have the analysis with the Liu Hu method which is very similar to the one we had before.

# # 3. Topic modelling

# ## Cleaning

# To get meaningful topics, we remove the words that occur in almost every e-mail and that do not convey information about the topic discussed in it (e.g words from the banner present in many of them: "unclassified u.s department of state...")

# In[22]:

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

# In[23]:

texts_tokenized3 = remove_unwanted_tokens(texts_tokenized2)


# Here is a comparison before and after removing unwanted tokens. <br />
# It seems like we lose many words, but it is actually only a few unique words appearing many times as well as some tokens smaller than 3 characters and the non-alphanumeric ones. This is fine as they do not add meaning to the e-mail. <br />
# The dictionnary we build later on has approximately 2000 tokens less after applying that function than before (from ~37500 tokens to ~35500 tokens)

# In[24]:

' '.join(texts_tokenized2[0])


# In[25]:

' '.join(texts_tokenized3[0])


# ## Corpus

# We create the dictionnary from the processed e-mail texts using gensim.corpora.dictionary.Dictionary 

# In[26]:

corpus_dictionary = Dictionary(texts_tokenized3)
print(corpus_dictionary)


# With that dictionnary we create a corpus made of the bag-of-words representation of each text

# In[27]:

corpus = [corpus_dictionary.doc2bow(tokens) for tokens in texts_tokenized3]


# ## Model

# Next, we train an Lda model on that corpus using gensim.models.LdaModel and the previously built dictionary. <br />
# We set the hyperparameters alpha and eta to 'auto' so that gensim manages and adapts them to our data. We also do 5 passes on the corpus to improve the correctness/accuracy of the topics.

# In[28]:

model = LdaModel(corpus, id2word=corpus_dictionary, num_topics=42, passes=5, alpha='auto', eta='auto', minimum_probability=0.02)


# In[29]:

model.save('best_model/best_model')


# In[30]:

model = LdaModel.load('best_model/best_model')


# We find that the topics the most understandable at first sight are the numbers 3, 41, 19, 16, 13, 18, 24, 8, 14, 36

# In[32]:

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

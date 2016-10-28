
# coding: utf-8

# # Funding choropleth map

# Some useful imports

# In[1]:

import requests as r
import pandas as pd
import json
import folium
import os
import re


# ## Data cleaning

# The files containing the funds of universities and the swiss cantons topology.

# In[2]:

funding_file = 'P3_GrantExport.csv'
cantons_file = 'ch-cantons.topojson.json'


# When reading the CSV file we select only the columns we are interested in, namely: "Project Number", "Institution", "University", and "Approved Amount"

# In[3]:

df = pd.read_csv(funding_file, sep=';', index_col=0, usecols=[0,6,7,13])


# In[4]:

df.head()


# We convert the elements of the "Approved Amount" column to numeric values, then we remove entries without a proper university name or with no readable amount.

# In[5]:

df['Approved Amount'] = pd.to_numeric(df['Approved Amount'], errors='coerce')


# In[6]:

df = df[(df.University != 'Nicht zuteilbar - NA') & pd.notnull(df.University) & pd.notnull(df['Approved Amount'])]


# In[7]:

df.head()


# ## Fill and compute

# We are using the google geocode API to get most of our canton information. To make this work without giving out any information on github you have to export a valid API key.

# In[8]:

wikimedia_cantons_url = 'https://fr.wikipedia.org/wiki/Canton_suisse'
geocoding_url = 'https://maps.googleapis.com/maps/api/geocode/json'
geocoding_key = os.environ.get('GEOCODING_KEY')
if geocoding_key is None:
    print('No key found: `export GEOCODING_KEY=...` before launching jupyter.')
    raise


# We gather all the cantons' acronyms and full names, from the topology file.

# In[9]:

cantons = []
with open(cantons_file) as file:    
    for canton in json.load(file)['objects']['cantons']['geometries']:
        cantons.append((canton['id'], canton['properties']['name'].split('/')))
assert len(cantons) == 26
cantons_acronyms = [c[0] for c in cantons]


# A list of all unique universities

# In[10]:

universities = df.University.unique()
universities


# This function will read the result of the google api and get out the canton initials back.

# In[11]:

def fetch_canton(value, clue, cantons, field='short_name', tpe='administrative_area_level_1', short_long_sep='-'):
        
    def req(address):
        return r.get(geocoding_url, params={'key': geocoding_key, 'address': address}).json()
    
    def extract(res):
        for location in res:
            canton = [elem[field] 
                      for elem in location['address_components'] 
                      if tpe in elem['types'] and elem[field] in cantons]
            if len(canton):
                return canton[0]
        return None
    
    full_query = req(value + ', ' + clue)
    if len(full_query['results']):
        return extract(full_query['results'])
    
    value_parts = value.split(short_long_sep)
    if len(value_parts) > 1:
        
        short_query = req(value_parts[1] + ', ' + clue)
        if len(short_query['results']):
            return extract(short_query['results'])
    
    return None


# First we do an heuristic search on a university name to see if it contains a canton's name, if not we make a request to Google's Geocoding API.

# In[12]:

universities_cantons = {}
for uni in universities:
    for acronym, names in cantons:
        for name in names:
            if name in uni:
                universities_cantons[uni] = acronym
    if not universities_cantons.get(uni):
        universities_cantons[uni] = fetch_canton(re.sub(r'\([^)]*\)', 'switzerland', uni), 'Switzerland', cantons_acronyms)


# The following code sends requests to Google's Geocoding API for all universities without a canton already found. The regex gets rid of the parenthesis when querying in order to ont flood the api with too much useless info.
# Also the function fetch_canton() adds a "clue", here it is "Switzerland", to the query.

# In[13]:

universities_cantons


# In[14]:

df['Cantons'] = df.University.apply(universities_cantons.get)


# Here is the percentage of universities we have located.

# In[15]:

pd.notnull(df.Cantons).sum() / len(df.Cantons)


# And the dataframe containing the canton linked to the institution.

# In[16]:

df.head()


# ## Preparing the Data

# The first part of visualizing the data is to get out how much investement per canton has be given. 

# In[17]:

df_funding = df.groupby('Cantons').sum().reindex(cantons_acronyms, fill_value=0).reset_index()
df_funding.describe()


# For the data to be put on a linear scale I apply the log function on the Approved amount to make it linear. The only thing special we do here is putting the 0 values (no financing amount) to a the min value so as to have as little outliers as possible.

# In[18]:

import numpy as np
df_funding["Approved Amount"][df_funding["Approved Amount"] == 0] = 1e+04
df_funding["Approved Amount"] = df_funding["Approved Amount"].apply(np.log)


# The describe below shows that the data is more usable for linear scale.

# In[19]:

df_funding.describe()


# ## Drawing

# Initialization of the map of switzerland and it's cantons.

# In[20]:

switzerland_center = 46.80111, 8.22667
switzerland_zoom = 8


# In[21]:

funding_map = folium.Map(location=switzerland_center, zoom_start=switzerland_zoom)


# In[22]:

funding_map.choropleth(
    geo_path=cantons_file,
    legend_name='logarithmic scale on the approved amounts',
    data=df_funding,
    columns=['Cantons', 'Approved Amount'],
    key_on='feature.id',
    topojson='objects.cantons',
    fill_color='YlGn',
    reset=True
)


# On the map we can very see the richest cantons and the poorer. We would have liked to add a legend to the scale as it is not really clear what the number means.

# We can also see the poorest cantons as the lightest ones thanks to the upgrade we gave them in the previous section.

# In[23]:

funding_map


# Saving the map in html for a bigger screen experience :).

# In[25]:

funding_map.save("./map.html")


# There is also an image in this folder.

# In[27]:

from IPython.display import Image
Image("./map.png")


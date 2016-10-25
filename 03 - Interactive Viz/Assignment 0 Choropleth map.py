
# coding: utf-8

# # Funding choropleth map

# In[1]:

import requests as r
import pandas as pd
import json
import folium
import os
import re


# ## Data cleaning

# In[2]:

funding_file = 'P3_GrantExport.csv'
cantons_file = 'ch-cantons.topojson.json'


# In[3]:

df = pd.read_csv(funding_file, sep=';', index_col=0, usecols=[0,6,7,13])


# In[4]:

df.head()


# In[5]:

df['Approved Amount'] = pd.to_numeric(df['Approved Amount'], errors='coerce')


# In[6]:

df = df[(df.University != 'Nicht zuteilbar - NA') & pd.notnull(df.University) & pd.notnull(df['Approved Amount'])]


# In[7]:

df.head()


# ## Fill and compute

# In[8]:

wikimedia_cantons_url = 'https://fr.wikipedia.org/wiki/Canton_suisse'
geocoding_url = 'https://maps.googleapis.com/maps/api/geocode/json'
geocoding_key = os.environ.get('GEOCODING_KEY')
if geocoding_key is None:
    print('No key found: `export GEOCODING_KEY=...` before launching jupyter.')
    raise


# In[9]:

cantons = []
with open(cantons_file) as file:    
    for canton in json.load(file)['objects']['cantons']['geometries']:
        cantons.append(canton['id'])
assert len(cantons) == 26
cantons


# In[10]:

universities = df.University.unique()
universities


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


# In[12]:

universities_cantons = {uni: fetch_canton(re.sub(r'\([^)]*\)', '', uni), 'Switzerland', cantons) for uni in universities}


# In[13]:

universities_cantons


# In[14]:

df['Cantons'] = df.University.apply(universities_cantons.get)


# In[15]:

pd.notnull(df.Cantons).sum() / len(df.Cantons)


# In[16]:

df.head()


# In[17]:

df_funding = df.groupby('Cantons').sum().reindex(cantons, fill_value=0).reset_index()
df_funding


# ## Drawing

# In[18]:

switzerland_center = 46.80111, 8.22667
switzerland_zoom = 8


# In[19]:

funding_map = folium.Map(location=switzerland_center, zoom_start=switzerland_zoom)


# In[20]:

funding_map.choropleth(
    geo_path=cantons_file, 
    data=df_funding,
    columns=['Cantons', 'Approved Amount'],
    key_on='feature.id',
    topojson='objects.cantons',
    fill_color='YlGn'
)


# In[21]:

funding_map


# ##  Röstigraben

# In[ ]:





# coding: utf-8

# # Gather Bachelor students data

# The purpose of this script is only to collect the data of Bachelor students, group and clean it all, serialize it in order to be used by other scripts analyzing the dataset.

# ## Assignment

# 1) Obtain all the data for the Bachelor students, starting from 2007. Keep only the students for which you have an entry for both Bachelor semestre 1 and Bachelor semestre 6.

# In[1]:

import requests as r
import pandas as pd

from bs4 import BeautifulSoup


# In[2]:

ISA_url = 'http://isa.epfl.ch/imoniteur_ISAP/!gedpublicreports'
ISA_html = ISA_url + '.html'
ISA_filter = ISA_url + '.filter'
ww_i_reportModel = '133685247'  # base reference from Postman
ww_i_reportModelXsl = '133685271'  # for XLS files from Postman


# ### Find the parameters values

# In order to find all the existing parameter names and possible values, we make a request on the form page and then parse it

# In[3]:

form_html = r.get(ISA_filter, params={'ww_i_reportModel': ww_i_reportModel})
form_bs = BeautifulSoup(form_html.content, "html.parser")


# We parse the HTML file looking for "select" tags which correspond to the different menus, and for each of them we look at the "option" tags which are the entries. We save our results in a dictionnary

# In[4]:

params = []

for select in form_bs.find_all('select'):
    param = select['name']
    
    for option in select.find_all('option'):
        name = option.string
        
        if name:
            value = option['value']
            params.append(pd.Series([param, name, value]))

params = pd.DataFrame(params)
params.columns = ['Parameter', 'Name', 'Value']
params.set_index(['Parameter', 'Name'], inplace=True)
params


# ### Gather the data

# In[5]:

def listOf(unite, year, semester):
    # retrieve gps reference using main form query for table access
    gps_html = r.get(ISA_filter, params={
        'ww_b_list': 1,
        'ww_i_reportmodel': ww_i_reportModel,
        'ww_i_reportModelXsl': ww_i_reportModelXsl,
        'ww_x_UNITE_ACAD': unite,
        'ww_x_PERIODE_ACAD': year,
        'ww_x_PERIODE_PEDAGO': semester
    })
    gps_bs = BeautifulSoup(gps_html.content, "html.parser")
    link = gps_bs.find_all('a')
    if len(link) != 2:
        return None
    
    # isolate second link (first one is always all) and rip off useless part
    gps = int(link[1]['onclick'][21:-16])
    table_html = r.get(ISA_html, params={
        'ww_i_reportmodel': ww_i_reportModel,
        'ww_i_reportModelXsl': ww_i_reportModelXsl,
        'ww_x_UNITE_ACAD': unite,
        'ww_x_PERIODE_ACAD': year,
        'ww_x_PERIODE_PEDAGO': semester,
        'ww_x_GPS': gps
    })
    table_bs = BeautifulSoup(table_html.content, "html.parser")
    tables = pd.read_html(str(table_bs), flavor='bs4', skiprows=[0, 1], header=0, index_col=10)
    
    if len(tables) != 1:
        return None
    
    table = tables[0]
    table.drop('Nom Prénom', axis=1, inplace=True)
    table.drop([col for col in table.columns if 'Unnamed' in col], axis=1, inplace=True)
    return table


# In this homework we are interesting in students from the "Informatique" section, and we want the data as excel files. Therefore we give the following parameters fixed values:

# In[6]:

unite = params.loc['ww_x_UNITE_ACAD', 'Informatique']
unite


# In[7]:

years = params.loc['ww_x_PERIODE_ACAD']
years


# In[8]:

param_semesters = params.loc['ww_x_PERIODE_PEDAGO']


# In[9]:

# do not take into account "b" semester as there are all empty (not sure of signification)
bachelor_semesters = param_semesters[['Bachelor' in idx and 'b' not in idx for idx in param_semesters.index]]
bachelor_semesters


# In[10]:

def gather(unite, years, semesters):
    yearly = []
    for year, year_value in years.iterrows():
        print(year)
        swarm = []
        for semester, semester_value in semesters.iterrows():
            print(semester)
            df = listOf(unite, year_value, semester_value)
            swarm.append(df)
        yearly.append(pd.concat(swarm, keys=semesters.index))
    return pd.concat(yearly, keys=years.index)


# In[11]:

bachelor = gather(unite, years, bachelor_semesters)


# In[12]:

bachelor.head()


# In[13]:

bachelor.to_pickle('bachelor')


# In[14]:

master_semesters = param_semesters[['Master' in idx or 'Projet' in idx for idx in param_semesters.index]]
master_semesters


# In[15]:

master = gather(unite, years, master_semesters)


# In[16]:

master.head()


# In[17]:

master.to_pickle('master')


# In[ ]:




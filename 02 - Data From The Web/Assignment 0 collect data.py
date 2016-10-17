
# coding: utf-8

# # Assignement 0 collect data from IS Academia

# The purpose of this notebook is only to collect the isa data over students, group and clean it all, serialize it in order to be used by other scripts analyzing the dataset.

# In[1]:

import requests as r
import pandas as pd

from bs4 import BeautifulSoup


# ## 1 General parameters

# Those data were taken out from Postman or simple network query analyzer.

# In[2]:

ISA_url = 'http://isa.epfl.ch/imoniteur_ISAP/!gedpublicreports'
ISA_html = ISA_url + '.html'
ISA_filter = ISA_url + '.filter'
ww_i_reportModel = '133685247'
ww_i_reportModelXsl = '133685271'


# ## 2 Dynamic parameters

# In order to find all the existing parameter names and possible values, we make a request on the form page and then parse it.

# In[3]:

form_html = r.get(ISA_filter, params={'ww_i_reportModel': ww_i_reportModel})
form_bs = BeautifulSoup(form_html.content, "html.parser")


# We parse the HTML file looking for `select` tags which correspond to the different menus, and for each of them we look at the `option` tags which are the entries.

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


# ## 3 Select parameters

# In this homework we are interesting in students from the "Informatique" section starting from 2007.

# In[5]:

unite = params.loc['ww_x_UNITE_ACAD', 'Informatique']
unite


# In[6]:

years = params.loc['ww_x_PERIODE_ACAD']
years


# In[7]:

param_semesters = params.loc['ww_x_PERIODE_PEDAGO']


# In[8]:

# do not take into account "b" semester as there are all empty (not sure of signification)
bachelor_semesters = param_semesters[['Bachelor' in idx and 'b' not in idx for idx in param_semesters.index]]
bachelor_semesters


# In[9]:

master_semesters = param_semesters[['Master' in idx or 'Projet' in idx for idx in param_semesters.index]]
master_semesters


# ## 4. Download data

# In[10]:

def listOf(unite, year, semester):
    # retrieve gps reference as if we submit isa form and got list of results
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
    
    # isolate second link (first one is always "Tous") and rip off useless part
    gps = int(link[1]['onclick'][21:-16])
    table_html = r.get(ISA_html, params={
        'ww_i_reportmodel': ww_i_reportModel,
        'ww_i_reportModelXsl': ww_i_reportModelXsl,
        'ww_x_UNITE_ACAD': unite,
        'ww_x_PERIODE_ACAD': year,
        'ww_x_PERIODE_PEDAGO': semester,
        'ww_x_GPS': gps
    })
    tables = pd.read_html(table_html.content, flavor='bs4', skiprows=[0, 1], header=0, index_col=10)
    if len(tables) != 1:
        return None
    
    # get the table and remove student names (avoid the name to be indexed by Google on Github...)
    table = tables[0]
    table.drop('Nom Prénom', axis=1, inplace=True)
    table.drop([col for col in table.columns if 'Unnamed' in col], axis=1, inplace=True)
    return table


# In[11]:

def gather(unite, years, semesters):
    # loop over years and semesters and assemble them into one data frame
    yearly = []
    for year, year_value in years.iterrows():
        print(year)
        swarm = []
        for semester, semester_value in semesters.iterrows():
            print(semester)
            df = listOf(unite, year_value, semester_value)
            swarm.append(df)
        yearly.append(pd.concat(swarm, keys=semesters.index))
    return pd.concat(yearly, keys=years.index, names=['Year', 'Semester', 'Sciper'])


# In[12]:

bachelor = gather(unite, years, bachelor_semesters)


# In[13]:

bachelor['Civilité'] = bachelor.Civilité.astype('category')
bachelor['Statut'] = bachelor.Statut.astype('category')
bachelor.head()


# In[14]:

master = gather(unite, years, master_semesters)


# In[15]:

master['Civilité'] = master.Civilité.astype('category')
master['Statut'] = master.Statut.astype('category')
master.head()


# ## 5 Serialize

# In[16]:

bachelor.to_pickle('bachelor')


# In[17]:

master.to_pickle('master')


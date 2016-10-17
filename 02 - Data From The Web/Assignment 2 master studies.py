
# coding: utf-8

# # Assignement 2 study of Master data from IS Academia

# In[1]:

import pandas as pd
import scipy.stats as stats
import numpy as np


# ## Import Master data

# Reading the pickle with the master data frame

# In[2]:

master_df = pd.read_pickle("master")
master_df.head()


# In[3]:

s1 = master_df.xs('Master semestre 1', level=1)
s1 = s1[(s1.Statut == 'Présent') & (s1.index.get_level_values(0) != '2016-2017')].reset_index(0)
print("number of students who started their semester since 2007 :", s1.shape[0])

sem2 = master_df.xs('Master semestre 2', level=1).loc[:,['Spécialisation', 'Mineur', 'Statut']].fillna("nan")
ma90 = master_df.xs('Master semestre 2', level=1)[sem2['Mineur'].str.contains("nan") & sem2['Spécialisation'].str.contains("nan")]
ma90 = sem2[(sem2.Statut == "Présent") & (sem2.index.get_level_values(0) != '2016-2017')].reset_index(0).index.drop_duplicates()
print("number of student ending the master with 90 cr :", ma90.shape[0])

sem3 = master_df.xs('Master semestre 3', level=1)
ma120 = sem3[(sem3.Statut == "Présent") & (sem3.index.get_level_values(0) != '2016-2017')].reset_index(0).index.drop_duplicates()
print("number of student ending the master with 120 cr or who did a 3rd semester :", ma120.shape[0])

master_sciper = s1.ix[s1.index & (ma90 | ma120)].index
print("number of student who finished their masters who started in 2007 : ", master_sciper.shape[0])

master_s1_papp = master_df.loc[master_df.index.get_level_values("Sciper").isin(master_sciper.values)]


# In[ ]:

master_s1_papp


# In[ ]:

master_semesters = []
scipers = master_s1_papp.index.get_level_values(2).drop_duplicates()

for sciper in scipers:
    student = master_s1_papp.xs(sciper, level=2)
    num_sems = student['Statut'].count() + 1
    print(student['Statut'])
    civ = student['Civilité'][0]
    spec = student['Spécialisation'][0]
    mineur = student['Mineur'][0]
    # If the student has neither a minor nor a spec, we write him as in no spé
    if str(mineur) == "nan" and str(spec) == "nan":
        spec = "aucun"
    
    master_semesters.append([sciper, civ, num_sems, mineur, spec])

master_semesters = pd.DataFrame(master_semesters,columns=['Sciper', 'Civilité', 'Number of semesters', 'Mineur', 'Spécialisation'])


# In[ ]:

master_semesters


# In[ ]:

spec_count = master_semesters.groupby('Spécialisation')['Spécialisation'].count()
sems_mean_spec = master_semesters.groupby('Spécialisation')['Number of semesters'].mean()
minor_count = master_semesters.groupby('Mineur')['Mineur'].count()
sems_mean_minor = master_semesters.groupby('Mineur')['Number of semesters'].mean()
sems_stats_spec = pd.concat([spec_count,sems_mean_spec], axis=1, keys=['Count','Mean number of semesters'])
sems_stats_minor = pd.concat([minor_count,sems_mean_minor], axis=1, keys=['Count','Mean number of semesters'])
sems_stats_tot = pd.concat([sems_stats_minor, sems_stats_spec], axis=0, keys=['mineur', 'spécialisation'])
print(master_semesters.median())
sems_stats_tot


# In[ ]:

import matplotlib.pyplot as plt
master_semesters.hist(bins=50)
plt.show()


# In[ ]:

total_mean = master_semesters['Number of semesters'].mean()
alpha = 0.05

for name, counts in master_semesters.groupby('Spécialisation')['Number of semesters']:
    test = stats.ttest_1samp(counts.values, total_mean)
    print("Results for {},".format(name), "the seperate means are :", counts.mean(), " and ", total_mean)
    print("Number of values :", len(counts.values))
    pvalue = test.pvalue
    
    if np.isnan(pvalue):
        print("not enough values to give a significant result")
    else:    
        print("Test p-value = %f" % pvalue)
        print("The means are statistically different") if test.pvalue < alpha else print("The means are statistically equal")
    print("")
        
        


# In[ ]:




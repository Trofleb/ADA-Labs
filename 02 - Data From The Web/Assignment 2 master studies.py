
# coding: utf-8

# # Assignement 2 study of Master data from IS Academia

# In[1]:

import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook')
get_ipython().magic('matplotlib inline')


# ## Import Master data

# Reading the pickle with the master data frame.

# In[2]:

master_df = pd.read_pickle("master")
master_df.head()


# Extract sciper and filter master by managing different cases: 90 ETCs master, 120 ETCs master (minor, etc.). We do not keep current year as not yet finished.

# In[3]:

s1 = master_df.xs('Master semestre 1', level=1)
s1 = s1[(s1.Statut == 'Présent') & (s1.index.get_level_values(0) != '2016-2017')].reset_index(0)
print("number of students who started their semester since 2007 :", s1.shape[0])

sem2 = master_df.xs('Master semestre 2', level=1).loc[:,['Spécialisation', 'Mineur', 'Statut']]
ma90 = master_df.xs('Master semestre 2', level=1)[sem2['Mineur'].isnull() & sem2['Spécialisation'].isnull()]
ma90 = sem2[(sem2.Statut == "Présent") & (sem2.index.get_level_values(0) != '2016-2017')].reset_index(0).index.drop_duplicates()
print("number of student ending the master with 90 cr :", ma90.shape[0])

sem3 = master_df.xs('Master semestre 3', level=1)
ma120 = sem3[(sem3.Statut == "Présent") & (sem3.index.get_level_values(0) != '2016-2017')].reset_index(0).index.drop_duplicates()
print("number of student ending the master with 120 cr or who did a 3rd semester :", ma120.shape[0])

master_sciper = s1.ix[s1.index & (ma90 | ma120)].index
print("number of student who finished their masters who started in 2007 : ", master_sciper.shape[0])

master_s1_papp = master_df.loc[master_df.index.get_level_values("Sciper").isin(master_sciper.values)]


# In[4]:

master_s1_papp.head()


# In[5]:

master_semesters = []
scipers = master_s1_papp.index.get_level_values(2).drop_duplicates()

for sciper in scipers:
    student = master_s1_papp.xs(sciper, level=2)
    num_sems = student['Statut'].count() + 1
    civ = student['Civilité'][0]
    spec = student['Spécialisation'][0]
    mineur = student['Mineur'][0]
    # If the student has neither a minor nor a spec, we write him as in no spec for future evaluation
    if str(mineur) == "nan" and str(spec) == "nan":
        spec = "aucun"
    
    master_semesters.append([sciper, civ, num_sems, mineur, spec])

master_semesters = pd.DataFrame(master_semesters,columns=['Sciper', 'Civilité', 'Number of semesters', 'Mineur', 'Spécialisation'])


# In[6]:

master_semesters.head(20)


# In[7]:

spec_count = master_semesters.groupby('Spécialisation')['Spécialisation'].count()
sems_mean_spec = master_semesters.groupby('Spécialisation')['Number of semesters'].mean()
minor_count = master_semesters.groupby('Mineur')['Mineur'].count()
sems_mean_minor = master_semesters.groupby('Mineur')['Number of semesters'].mean()
sems_stats_spec = pd.concat([spec_count,sems_mean_spec], axis=1, keys=['Count','Mean number of semesters'])
sems_stats_minor = pd.concat([minor_count,sems_mean_minor], axis=1, keys=['Count','Mean number of semesters'])
sems_stats_tot = pd.concat([sems_stats_minor, sems_stats_spec], axis=0, keys=['mineur', 'spécialisation'])
print(master_semesters.median())
sems_stats_tot


# In[8]:

import matplotlib.pyplot as plt
master_semesters.hist(bins=50)
plt.show()


# In[9]:

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
    print()


# ## Bonus: Statistical difference between men and women ?

# We separate the data between men and women, computet the means and test if they are statistically different the same way we used for bachelor stduents.

# In[10]:

master_semesters
gender_count = master_semesters.groupby('Civilité')['Civilité'].count()
sems_mean = master_semesters.groupby('Civilité')['Number of semesters'].mean()
sems_stats = pd.concat([gender_count,sems_mean], axis=1, keys=['Count','Mean number of semesters'])
sems_stats


# The means are very close to each other, the following "Two-Sample T-Test" shows it by having a high p-value. Therefore it cannot rejecting the null hyptohesis being that the means are similar.

# In[11]:

male_sems_num = master_semesters.query('Civilité == "Monsieur"')['Number of semesters'].values
female_sems_num = master_semesters.query('Civilité == "Madame"')['Number of semesters'].values

alpha = 0.05
test = stats.ttest_ind(male_sems_num, female_sems_num, equal_var=False)

print("p-value = %f" % test.pvalue)
print("statistically " + ("different" if test.pvalue < alpha else "similar"))


# ## Plotting

# Add last year of Master for each student

# In[12]:

last_years = []
for sciper in master_sciper:
    years = master_s1_papp.xs(sciper, level=2).iloc[:1]
    last_year = years.index.get_level_values(0).tolist()[0]
    last_years.append([sciper,last_year])

tmp = pd.DataFrame(last_years, columns=['Sciper', 'Last year'])
master_semesters = master_semesters.merge(tmp, left_on='Sciper', right_on='Sciper')
master_semesters.head()


# In[13]:

master_semesters['Mineur'] = master_semesters['Mineur'].astype('category')
master_semesters['Spécialisation'] = master_semesters['Spécialisation'].astype('category')


# In[14]:

master_years = master_semesters.groupby(['Last year'])


# We plot the repartition of the number of semesters each student has done acording to the year they finished and their mineur/spécialisation. One assumption of statistically similar tests is to see female and male points mixed, without clear distinction in terms of semester distribution.

# In[15]:

years = master_s1_papp.index.get_level_values(0).drop_duplicates()
for y in years:
    students = master_years.get_group(y)
    
    m = students.query('Civilité == "Monsieur"')[['Sciper', 'Number of semesters', 'Mineur', 'Spécialisation']]
    f = students.query('Civilité == "Madame"')[['Number of semesters', 'Mineur', 'Spécialisation']]
    
    m_sems = m['Number of semesters'].values
    f_sems = f['Number of semesters'].values
    
    m_mineur = m['Mineur'].cat.codes
    f_mineur = f['Mineur'].cat.codes
    
    m_spe = m['Spécialisation'].cat.codes
    f_spe = f['Spécialisation'].cat.codes
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.set_title(y)
    ax1.scatter(m_mineur, m_sems, c='blue', label='men')
    ax1.scatter(f_mineur, f_sems, c='red', label='women')
    ax1.set_xlabel('Mineur')
    ax1.set_ylabel('Num semesters')
    ax1.legend(loc='best')
    
    ax2.set_title(y)
    ax2.scatter(m_spe, m_sems, c='blue', label='men')
    ax2.scatter(f_spe, f_sems, c='red', label='women')
    ax2.set_xlabel('Spécialisation')
    ax2.legend(loc='best')


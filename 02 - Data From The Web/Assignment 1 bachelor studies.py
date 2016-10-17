
# coding: utf-8

# # Assignement 1 study of Bachelor data from IS Academia

# In[1]:

import pandas as pd
import numpy as np
import scipy.stats as stats


# ## Import Bachelor data

# Reading the pickle with the bachelor data frame.

# In[2]:

bachelor_df = pd.read_pickle("bachelor")
bachelor_df.head()


# All students that did semester 1 and 6, extract sciper and filter bachelor to keep only those students. A finishing student can be either "Présent" or in echange during last year.

# In[3]:

s1 = bachelor_df.xs('Bachelor semestre 1', level=1)
s1 = s1[s1.Statut == 'Présent'].reset_index(0)

s6 = bachelor_df.xs('Bachelor semestre 6', level=1)
s6 = s6[(s6.Statut == 'Présent') | (s6['Type Echange'] != np.nan)].reset_index(0)

sciper_s1_s6 = s1.ix[s1.index & s6.index].index
bachelor_s1_s6 = bachelor_df.loc[bachelor_df.index.get_level_values("Sciper").isin(sciper_s1_s6.values)]

bachelor_s1_s6.head()


# ## Compute the number and the mean of semesters spent for the Bachelor by gender

# Let's find how many semesters each student spent for to get its Bachelor. To do we iterate over all the scipers and count the number of status (that includes semesters in Erasmus exchange and 'off semesters' as well).

# In[4]:

bachelor_semesters = []
scipers = bachelor_s1_s6.index.get_level_values(2).drop_duplicates()

for sciper in scipers:
    student = bachelor_s1_s6.xs(sciper, level=2)
    num_sems = student['Statut'].count()
    civ = student['Civilité'][0]
    bachelor_semesters.append([sciper, civ, num_sems])

bachelor_semesters = pd.DataFrame(bachelor_semesters,columns=['Sciper', 'Civilité', 'Number of semesters'])


# We now group the results by the 'Civilité' and average the number of semesters.

# In[5]:

gender_count = bachelor_semesters.groupby('Civilité')['Civilité'].count()
sems_mean = bachelor_semesters.groupby('Civilité')['Number of semesters'].mean()
sems_stats = pd.concat([gender_count,sems_mean], axis=1, keys=['Count','Mean number of semesters'])
sems_stats


# ## Perform a "hypothesis testing" test

# Our null hypothethis H0 is that the difference between the two means is statistically insignificant. Using "Two-Sample T-Test" based on our two sample population, we select our alpha to be 0.05 in order to be sure at 95% with no assumption concerning the variance.

# In[6]:

male_sems_num = bachelor_semesters.query('Civilité == "Monsieur"')['Number of semesters'].values
female_sems_num = bachelor_semesters.query('Civilité == "Madame"')['Number of semesters'].values

alpha = 0.05
test = stats.ttest_ind(male_sems_num, female_sems_num, equal_var=False)

print("p-value = %f" % test.pvalue)
print("statistically " + ("different" if test.pvalue < alpha else "similar"))


# Since our test p-value is greater than the alpha chosen we cannot reject our null hypothesis, meaning that both mean are statistically identicals.
# One caveat in this test is the population of female students (24) which is pretty low.

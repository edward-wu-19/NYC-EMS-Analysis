#%% Basic Information Header
'''
@author Bince
'''
#%% Importing Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#%% Loading Data
# this data set has already been filtered on the website I got this from, I custom tailored
# to only include 10 rows, that of which are incident datetime, intial call type, final severity
# level code, first assignment datetime, dispatch response in seconds, incident response in seconds,
# incident travel time in seconds, the borough, the incident dispatch area and the community district
# I also made queried to remove NaNs on the website from the community district  column as well as 
# only keeping emergency calls that occurred in the year 2021. This was done because my median income
# data is only based on 2021. 
df_EMS_all = pd.read_csv('EMS_Incident_Dispatch_Data_2021.csv')

# New Dataset we are reading form, previous one is irrelevant now
# this dataset is also only based on 2021
df_codedIncomes = pd.read_csv('Median Income.csv')

def commCodeConversion(df):
    tmp = []
    
    for code in df['COMMUNITYDISTRICT']:
        codeStr = str(code)
        
        # replacement function, the third argument indicates how many of the first occurences 
        # should be replaced, thus 1 is specified in order to replace the first instance only
        if codeStr[0] == '1':
            codeStr = codeStr.replace('1', "M", 1)
        elif codeStr[0] == '2':
            codeStr = codeStr.replace('2', "B", 1)
        elif codeStr[0] == '3':
            codeStr = codeStr.replace('3', "K", 1)
        elif codeStr[0] == '4':
            codeStr = codeStr.replace('4', "Q", 1)
        elif codeStr[0] == '5':
            codeStr = codeStr.replace('5', "S", 1)       
        
        tmp.append(codeStr)
    return tmp
        
changedCodes = commCodeConversion(df_EMS_all)
df_EMS_all['New_Community_District'] = changedCodes

#%% Sanity Check to Make Sure Community District codes and Neighborhood Code are the same

df_EMS_codes = set(df_EMS_all['New_Community_District'])
df_income_codes = set(df_codedIncomes['Neighborhood Code'])

overlap = df_EMS_codes.intersection(df_income_codes)

# these extra codes most likely refer to parts of NYC that are entirely made up of large parks
# or protected wildlife areas, which is why they don't show up in the income data because
# no one really lives there: https://en.wikipedia.org/wiki/Community_boards_of_New_York_City
# this link helped resolve the issue and its description of the community codes definitely
# align with the income data regarding the number of community districts within each borough
# so these non-overlap codes are probably emergency calls that happened in parks, which makes sense
non_overlap = df_EMS_codes.difference(overlap)

code_list = list(overlap)
code_list.sort()

# we only lose 3909 rows
df_EMS = df_EMS_all[df_EMS_all['New_Community_District'].isin(code_list)]
df_severe = df_EMS[['FINAL_SEVERITY_LEVEL_CODE', 'New_Community_District']]

# descriptive statistics
means = df_severe.groupby('New_Community_District').mean()
medians = df_severe.groupby('New_Community_District').median()
stds = df_severe.groupby('New_Community_District').std()

# reset their indexes and column names
means = means.reset_index()
means = means.rename(columns={"FINAL_SEVERITY_LEVEL_CODE":"Mean Severity Code"})

medians = medians.reset_index()
medians = medians.rename(columns={"FINAL_SEVERITY_LEVEL_CODE":"Median Severity Code"})

stds = stds.reset_index()
stds = stds.rename(columns={"FINAL_SEVERITY_LEVEL_CODE":"Std Severity Code"})

# specific income data
df_income = df_codedIncomes[['Community Districts', 'Neighborhood Code', 'All Households']]
df_income = df_income.sort_values('Neighborhood Code')
df_income = df_income.reset_index()
df_income = df_income.drop(columns=['index'])

# combine!
df_income = df_income.merge(means, left_on='Neighborhood Code', right_on='New_Community_District')
df_income = df_income.drop(columns='New_Community_District')

df_income = df_income.merge(medians, left_on='Neighborhood Code', right_on='New_Community_District')
df_income = df_income.drop(columns='New_Community_District')

df_income = df_income.merge(stds, left_on='Neighborhood Code', right_on='New_Community_District')
df_income = df_income.drop(columns='New_Community_District')

#%% Clean up in the income aisle
print(df_income.dtypes)

# convert the object to a beautiful float
tmp = []
for inc in df_income['All Households']:
    inc = inc.replace('$', "").replace(',',"")
    inc = float(inc)
    tmp.append(inc)
    
df_income['Household Income'] = tmp


r = stats.pearsonr(df_income['Household Income'], df_income['Mean Severity Code']).correlation
plt.scatter(df_income['Household Income'], df_income['Mean Severity Code'], marker='.')
plt.xlabel("Average Income of A Neighborhood")
plt.ylabel("Mean Severity Code")
plt.title("A Neighborhood's Mean Severity of an Emergency Call versus Their Income 2021")
plt.show()

# relatively weak correlation to show that as a neighborhood's mean income increases, the severity
# of a 911 call decreases. maybe it might help to sort it out by borough

df_queens = df_income[df_income['Neighborhood Code'].str.contains('Q')]
df_brooklyn = df_income[df_income['Neighborhood Code'].str.contains('K')]
df_bronx = df_income[df_income['Neighborhood Code'].str.contains('B')]
df_staten = df_income[df_income['Neighborhood Code'].str.contains('S')]
df_manhattan = df_income[df_income['Neighborhood Code'].str.contains('M')]

rQueens = stats.pearsonr(df_queens['Household Income'], df_queens['Mean Severity Code']).correlation
rBrooklyn = stats.pearsonr(df_brooklyn['Household Income'], df_brooklyn['Mean Severity Code']).correlation
rBronx = stats.pearsonr(df_bronx['Household Income'], df_bronx['Mean Severity Code']).correlation
rStaten = stats.pearsonr(df_staten['Household Income'], df_staten['Mean Severity Code']).correlation
rManhattan = stats.pearsonr(df_manhattan['Household Income'], df_manhattan['Mean Severity Code']).correlation

label = ['Queens', 'Brooklyn', 'Bronx', 'Staten Island', 'Manhattan']
plt.scatter(df_bronx['Household Income'], df_bronx['Mean Severity Code'], marker='.')
plt.scatter(df_brooklyn['Household Income'], df_brooklyn['Mean Severity Code'], marker='.')
plt.scatter(df_manhattan['Household Income'], df_manhattan['Mean Severity Code'], marker='.')
plt.scatter(df_queens['Household Income'], df_queens['Mean Severity Code'], marker='.')
plt.scatter(df_staten['Household Income'], df_staten['Mean Severity Code'], marker='.')
plt.legend(label)
plt.xlabel("District Average Household Income (USD)")
plt.ylabel("Mean Severity Code")
plt.title("District Income vs. Mean Call Severity")
plt.show()

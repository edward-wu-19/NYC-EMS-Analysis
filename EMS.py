#%% Basic Information Header
'''
@author: Vincent Qiu
@date-created: January 12th, 2025
'''
#%% Importing Necessary Packages
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
#%% Load Data
# this data set has already been filtered on NYC Open Data itself, I custom tailored it
# to only include 5 columns, that of which are 'INCIDENT_DATETIME', 'FINAL_SEVERITY_LEVEL_CODE', 'BOROUGH',
# 'ZIPCODE', and 'COMMUNITYDISTRICT'. Additionally, I only kept emergency calls that occurred in the year 2021. 
# This was done because my median income data is only based on 2021. So now we will load it: 
df_EMS_all = pd.read_csv('./Downloads/EMS_Incident_Dispatch_Data_2021.csv')

# filter out NaNs
df_EMS_all = df_EMS_all.dropna()

# this data set comes from cccnewyork.org, I took the data it displayed and turned into a csv which 
# means that NaNs have already been filtered out entirely. Now we will load the income dataset:
df_codedIncomes = pd.read_csv('./Downloads/Median Income.csv')

#%% Converting Community District Code in the EMS Dataset to Match the Neighborhood District codes from the Income Data 
# Queens = 4, Bronx = 2, Manhattan = 1, Brooklyn = 3, Staten Island = 5
def commCodeConversion(df):
    tmp = []
    
    for code in df['COMMUNITYDISTRICT']:
        codeStr = str(int(code))
        
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
        
        # replace with changed code
        tmp.append(codeStr)
    return tmp
        
# create a new column containing these reformatted codes
changedCodes = commCodeConversion(df_EMS_all)
df_EMS_all['New_Community_District'] = changedCodes

#%% Sanity Check to Ensure Community District Codes and Neighborhood Code are the Same

# create a set from the codes, this will leave only unique values
df_EMS_codes = set(df_EMS_all['New_Community_District'])
df_income_codes = set(df_codedIncomes['Neighborhood Code'])

# check to see which codes overlap, this will help us determine if there are codes that don't match
# which will help us ensure whether or not the two columns label the same districts and neighborhoods
overlap = df_EMS_codes.intersection(df_income_codes)

# these extra codes below most likely refer to parts of NYC that are entirely made up of large parks
# or protected wildlife areas, which is why they don't show up in the income data because
# no one really lives there: https://en.wikipedia.org/wiki/Community_boards_of_New_York_City
# this link helped resolve the issue and its description of the community codes definitely
# align with the income data regarding the number of community districts within each borough
# so these non-overlap codes are probably emergency calls that happened in parks, which makes sense
non_overlap = df_EMS_codes.difference(overlap)

# create a list to use as a key to sort through the df_EMS_all dataset
code_list = list(overlap)
code_list.sort()

# we only lose 3909 rows
df_EMS = df_EMS_all[df_EMS_all['New_Community_District'].isin(code_list)]

# next we will find the mean, median, and standard deviation of each neighborhood
df_severe = df_EMS[['FINAL_SEVERITY_LEVEL_CODE', 'New_Community_District']]

# descriptive statistics grouped by each community district
# this will help us find the mean, median, and standard deviation severity level code based on each neighborhood 
means = df_severe.groupby('New_Community_District').mean()
medians = df_severe.groupby('New_Community_District').median()
stds = df_severe.groupby('New_Community_District').std()

# reset the mean column's index and its name
means = means.reset_index()
means = means.rename(columns={"FINAL_SEVERITY_LEVEL_CODE":"Mean Severity Code"})

# reset the median column's index and its name
medians = medians.reset_index()
medians = medians.rename(columns={"FINAL_SEVERITY_LEVEL_CODE":"Median Severity Code"})

# reset the standard deviation column's index and its name
stds = stds.reset_index()
stds = stds.rename(columns={"FINAL_SEVERITY_LEVEL_CODE":"StDev Severity Code"})

# format our income data
df_income = df_codedIncomes[['Community Districts', 'Neighborhood Code', 'All Households']]
df_income = df_income.sort_values('Neighborhood Code')
df_income = df_income.reset_index()
df_income = df_income.drop(columns=['index'])

# add on our mean severity level code column to the final dataframe we will be using
df_income = df_income.merge(means, left_on='Neighborhood Code', right_on='New_Community_District')
df_income = df_income.drop(columns='New_Community_District')

# add on our median severity level code column to the final dataframe we will be using
df_income = df_income.merge(medians, left_on='Neighborhood Code', right_on='New_Community_District')
df_income = df_income.drop(columns='New_Community_District')

# add on our standard deviation column to the final dataframe we will be using
df_income = df_income.merge(stds, left_on='Neighborhood Code', right_on='New_Community_District')
df_income = df_income.drop(columns='New_Community_District')

#%% Clean up in the Income Column
print(df_income.dtypes)

# convert the column of object types to a proper one: a float type
tmp = []
for inc in df_income['All Households']:
    inc = inc.replace('$', "").replace(',',"")
    inc = float(inc)
    tmp.append(inc)

# create a new column containing the properly formatted data
df_income['Household Income'] = tmp

# calculate the correlation between a neighborhood's mean severity code and its average household income across NYC
r = stats.pearsonr(df_income['Household Income'], df_income['Mean Severity Code']).correlation

# graph the relationship between a neighborhood's mean severity code and its average household income
plt.scatter(df_income['Household Income'], df_income['Mean Severity Code'], marker='.')
plt.xlabel("Average Income of A Neighborhood")
plt.ylabel("Mean Severity Code")
plt.title("A Neighborhood's Mean Severity of an Emergency Call versus Their Income 2021")
plt.show()

# there's a relatively weak but positive correlation between a neighborhood's average household income
# and its average severity of an emergency. However, perhaps there's a different pattern underlying
# if we filter this data by borough. 
df_queens = df_income[df_income['Neighborhood Code'].str.contains('Q')]
df_brooklyn = df_income[df_income['Neighborhood Code'].str.contains('K')]
df_bronx = df_income[df_income['Neighborhood Code'].str.contains('B')]
df_staten = df_income[df_income['Neighborhood Code'].str.contains('S')]
df_manhattan = df_income[df_income['Neighborhood Code'].str.contains('M')]

# calculate the correlation between a neighborhood's household income and its mean severity code for each borough
rQueens = stats.pearsonr(df_queens['Household Income'], df_queens['Mean Severity Code']).correlation
rBrooklyn = stats.pearsonr(df_brooklyn['Household Income'], df_brooklyn['Mean Severity Code']).correlation
rBronx = stats.pearsonr(df_bronx['Household Income'], df_bronx['Mean Severity Code']).correlation
rStaten = stats.pearsonr(df_staten['Household Income'], df_staten['Mean Severity Code']).correlation
rManhattan = stats.pearsonr(df_manhattan['Household Income'], df_manhattan['Mean Severity Code']).correlation

# graph our neighborhoods separated by borough
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

#%% Significance Testing
# ANOVA test to determine if there's a significant difference in the mean severity code across neighborhoods
# that are low income vs medium income vs high income

# let's cut our data into quartiles: 
    # low income are neighborhoods in the bottom 25 percentile, 
    # medium income is anything between 25% and 75% percentiles
    # high income will be any range above the 75 percentile

# run summary statistics on our income column
summ_stats = df_income['Household Income'].describe()
print("These are our percentiles:")
print(summ_stats[4:7])

# split the data into our respective groups
df_LI = df_income[df_income["Household Income"] <= summ_stats.iloc[4]]
df_MI = df_income[df_income["Household Income"] > summ_stats.iloc[4]]
df_MI = df_MI[df_MI['Household Income'] < summ_stats.iloc[6]]
df_HI = df_income[df_income["Household Income"] >= summ_stats.iloc[6]]

# before performing the anova test, let's state the hypotheses:

    # Null Hypothesis: the severity code means between the low, medium, and high income neighborhoods are
    # approximately equal
    
    # Alternative Hypothesis: the severity code mean of at least one of the three income groups differ

# time to perform the anova test....
results = f_oneway(df_LI['Mean Severity Code'], df_MI['Mean Severity Code'], df_HI['Mean Severity Code'])
# get p-value and interpret it
p = results[1]
if p < 0.05:
    print("Our data is statistically significant indicating that the chance of yielding this data by chance alone is unlikely. "
              + "Thus, we can reject the null hypothesis and claim that at least one of the neighborhood income groups have a different average emergency severity.")
else:
    print("Our data is not statistically significant, indicating that we can not rule out the possibility of observing this data by chance alone. "
              + "Thus, we fail to reject the null hypothesis.")

# plotting the graph that showcases the mean emergency severity of a neighborhood and the incomes
plt.scatter(df_LI['Household Income'], df_LI['Mean Severity Code'], marker='.')
plt.scatter(df_MI['Household Income'], df_MI['Mean Severity Code'], marker='.')
plt.scatter(df_HI['Household Income'], df_HI['Mean Severity Code'], marker='.')
plt.legend(['Low Income', 'Medium Income', 'High Income'])
plt.xlabel("District Average Household Income (USD)")
plt.ylabel("Mean Severity Code")
plt.title("District Income vs. Mean Call Severity")
plt.show()
# the plot helps us understand the data more but also leaves questions stil, I will try a different 
# significance test between solely between the low income and high income neighborhoods.

#%% Welch t-test Between High Income and Low Income Neighborhoods

# Units of Analysis: NYC neighborhoods 

# Assumption 1: I will assume that the variability between each individual neighborhood within each of the income groups 
# isn't very large. Essentially, I assume that a low income neighborhood doesn't vary greatly from another low income neighborhood or 
# that a high income neighborhood doesn't vary greatly from another high income neighborhood

# Assumption 2: The variation between low income and high income groups cannot be the same, this is due to the
# distribution of the graph we created after the ANOVA test. Thus, I will assume the variations between each
# income group is not the same. This is also further evidenced by each group's standard deviation.
# Since the homogeneity of variances between groups is violated, we must perform a Welch t-test.

# test to see if homoegeneity of variance holds:
LI_std = df_LI["Mean Severity Code"].std()
HI_std = df_HI["Mean Severity Code"].std()
# HI_std = 0.1057 while LI_std = 0.0681, as we can see these standard deviations are very different
# which violates homogeneity variance

# next, before performing the Welch t-test, let's state the hypotheses:
    
    # Null Hypothesis: the mean severity between the low and high income neighborhoods are equal
    
    # Alternative Hypothesis: the mean severity in the low income neighborhood is higher than the mean severity
    # in the high income neighborhood

# now perform the Welch t-test
welch_results = stats.ttest_ind(df_LI['Mean Severity Code'], df_HI['Mean Severity Code'], equal_var=False, alternative='greater')

# grab our p-value and interpret it
p_welch = welch_results[1]
if p_welch < 0.05:
    print("Our data is statistically significant, indicating it is unlikely we observed this data due to chance alone. "
              + "Thus, we can reject the null hypothesis and claim that the mean severity in the low income neighborhood is higher than "
              + "the mean severity in the high income neighborhood.")
else:
    print("Our data is not statistically significant, indicating that we can not rule out the possibility of observing this data by chance alone. "
              + "Thus, we fail to reject the null hypothesis.")
    
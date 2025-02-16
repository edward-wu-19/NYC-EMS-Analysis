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
#%% \\ IRRELEVANT // Analyze that damn fuckin' dataset
# df_EMS = df_EMS_all[['INCIDENT_DATETIME', 'FINAL_SEVERITY_LEVEL_CODE', 'COMMUNITYDISTRICT', 'ZIPCODE']]
#%% Loading Other Reasonable Data
# df_inc_unclean = pd.read_csv('Median_Incomes_in_New_York_by_District.csv')
# # A district's average income should include all households... so what do i care
# df_incomes_unclean = df_inc_unclean[df_inc_unclean['Household Type'] == "All Households"]
# df_income = df_incomes_unclean[df_incomes_unclean['TimeFrame'] == 2021]

# New Dataset we are reading form, previous one is highkey irrelevant now
# this dataset is also only based on 2021 so
df_codedIncomes = pd.read_csv('Median Income.csv')
#%% \\ IRRELEVANT // NEXT: create a function that adds a new column to df_EMS where it assigns the district name based on the zipcode
# big problem: multiple neighborhoods will have to be combined simply because certain zipcodes are indistinguishable
# from either neighborhood, so to assign a specific neighborhood would be really unfair and any of the precinct 
# standards / codes aren't based on the neighborhood and spill into any neighborhood every which way

# placeholder variable to insert into the column
# plhld = len(df_EMS) * ['NaN']
# df_EMS['Neighborhood'] = plhld

# zipcodeData = {'Neighborhood':['Kingsbridge - Riverdale', 'Northeast Bronx',
#                           'Fordham - Bronx Park', 'Pelham - Throgs Neck',
#                           'Crotona - Tremont', 'High Bridge - Morrisania',
#                           'Hunts Point - Mott Haven', 'Greenpoint',
#                           'Downtown - Heights - Park Slope',
#                           'Bedford Stuyvesant - Crown Heights', 'East New York',
#                           'Sunset Park', 'Borough Park', 'East Flatbush - Flatbush',
#                           'Canarsie - Flatlands', 'Bensonhurst - Bay Ridge',
#                           'Coney Island - Sheepshead Bay', 'Williamsburg - Bushwick',
#                           'Washington Heights - Inwood',
#                           'Central Harlem - Morningside Heights', 'East Harlem',
#                           'Upper West Side', 'Upper East Side', 'Chelsea - Clinton',
#                           'Gramercy Park - Murray Hill', 'Greenwich Village - SoHo',
#                           'Union Square - Lower East Side', 'lower Manhattan',
#                           'Long Island City - Astoria', 'West Queens',
#                           'Flushing - Clearview', 'Bayside - Little Neck', 
#                           'Ridgewood - Forest Hills', 'Fresh Meadows',
#                           'Southwest Queens', 'Jamaica', 'Southeast Queens',
#                           'Rockaway', 'Port Richmond', 'Stapleton - St. George',
#                           'Willowbrook', 'South Beach - Tottenville'],
#           'Zipcodes':[[10463, 10471],[10466, 10469, 10470, 10475], [10458, 10467, 10468],
#                       [10461, 10462, 10464, 10465, 10472, 10473], [10453, 10457, 10460],
#                       [10451, 10452, 10456], [10454, 10455, 10459, 10474], [11211, 11222],
#                       [11201, 11205, 11215, 11217, 11231], [11213, 11212, 11216, 11233, 11238],
#                       [11207, 11208], [11220, 11232], [11204, 11218, 11219, 11230],
#                       [11203, 11210, 11225, 11226], [11234, 11236, 11239], [11209, 11214, 11228],
#                       [11223, 11224, 11229, 11235], [11206, 11221, 11237],
#                       [10031, 10032, 10033, 10034, 10040], [10026, 10027, 10030, 10037, 10039],
#                       [10029, 10035], [10023, 10024, 10025], [10021, 10028, 10044, 10128],
#                       [10001, 10011, 10018, 10019, 10020, 10036], [10010, 10016, 10017, 10022],
#                       [10012, 10013, 10014], [10002, 10003, 10009], [10004, 10005, 10006, 10007, 10038, 10280],
#                       [11101, 11102, 11103, 11104, 11105, 11106], [11368, 11369, 11370, 11372, 11373, 11377, 11378],
#                       [11354, 11355, 11356, 11357, 11358, 11359, 11360], [11361, 11362, 11363, 11364],
#                       [11374, 11375, 11379, 11385], [11365, 11366, 11367],
#                       [11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421],
#                       [11412, 11423, 11432, 11433, 11434, 11435, 11436],
#                       [11004, 11005, 11411, 11413, 11422, 11426, 11427, 11428, 11429],
#                       [11691, 11692, 11693, 11694, 11695, 11697], [10302, 10303, 10310], [10301, 10304, 10305],
#                       [10314], [10306, 10307, 10308, 10309, 10312]]}

# # create the dataframe
# df_KEY = pd.DataFrame(zipcodeData)
#%% \\ IRRELEVANT // Conversion function

# def zipcode_conversion(df1, df2):
#     for i in range(len(df2)):
#         # create a mask that sorts by rows where one of the row value matches at least one of the given keys
#         # ie: if a neighborhood has the zipcodes 1,2,3,4 then it will filter the df1 zipcodes if any
#         # of those zipcodes match either 1,2,3 or 4
#         mask = df1['ZIPCODE'].isin(df2['Zipcodes'].loc[i])
        
#         # replace the selected rows with the proper neighborhood name as it was previously set to NaN
#         df1.loc[mask, 'Neighborhood'] = df2['Neighborhood'].loc[i]

# # run the function
# zipcode_conversion(df_EMS, df_KEY)

#%% \\ IRRELEVANT // Find overlapping codes with the income data and dispatch data
# # first, grab all the codes that exist within the EMS data
# df_districtCode = df_EMS_all['INCIDENT_DISPATCH_AREA']

# # next turn it into the set to get unique ones
# dcSet = set(df_districtCode)

# # do the same for the income data
# dcI = set(df_codedIncomes['Neighborhood Code'])

# # turn them into lists and sort them to make it easier to see
# dcSet = list(dcSet)
# dcI = list(dcI)

# dcSet.sort()
# dcI.sort()

# # turn this one into a dataframe in order to keep track of the original codes 
# # this will be necessary when sorting through the income data set with the specific codes that overlap
# # this is because the two datasets use the same codes formatted differently so exact string queries yield nothing
# df_keyInc = pd.DataFrame(dcI)

# # do the reformatting
# # the median income format prefers to have 3 letter/digit codes so a 0 is always
# # inserted into a code when the number is a single digit. For example, M1 is 
# # written as M01 to keep the 1 letter 2 digit format, but M12 is kept as M12 
# # thus to correct this, we will remove every 0 from the second position of the 
# # dispatch code if it exists to convert it properly
# for j in range(len(dcI)):
#     cStr = dcI[j]
#     if cStr[1] == '0':
#         cStr = cStr.replace('0', "")
#     dcI[j] = cStr

# # add the reformatted data that's still in the same order
# df_keyInc['1'] = dcI
    
# # convertt back into sets so we may do the intersection of the two code groups
# dcSet = set(dcSet)
# dcI = set(dcI)
    
# # convert our intersected unique set into a list to use as a filter
# overlappingDispatchCodes = dcSet.intersection(dcI)
# overlappingDispatchCodes = list(overlappingDispatchCodes)

#%% \\ IRRELEVANT // Sorting hat for specific district codes
# this functions the  same as our function earlier but im too lazy to make it universal
# it grabs the EMS calls that correspond to the specific dispatch codes
# mask = df_EMS['INCIDENT_DISPATCH_AREA'].isin(overlappingDispatchCodes)
# df_final = df_EMS[mask]

# # this grabs the unique keys from our data that match the specific dispatch codes
# # this is necessary to create the filter for the income data as overlapping dispatch codes doesn't work
# mask = df_keyInc['1'].isin(overlappingDispatchCodes)
# df_finKey = df_keyInc[mask]

# # now we extract our filter and convert it into list format, then we use it to extract the corresponding codes
# correctedODC = df_finKey[0]
# correctedODC = list(correctedODC)
# mask = df_codedIncomes['Neighborhood Code'].isin(correctedODC)
# df_finalInc = df_codedIncomes[mask]

#%% \\ IRRELEVANT // Fixing the dates kms.

# true_false = []

# for date in df_final['INCIDENT_DATETIME']:
#     if '2021' in date:
#         true_false.append(True)
#     else:
#         true_false.append(False)
        
# df_final['is from 2021?'] = true_false

# mask = df_final['is from 2021?'] == True
# df_finality = df_final[mask]      

#%% Converting community district code to match the neighborhood district codes from the income data 
# Queens = 4, Bronx = 2, Manhattan = 1, Brooklyn = 3, Staten Island = 5
print(df_EMS_all.dtypes)

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
        
        # replace with changed code
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
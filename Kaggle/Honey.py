import numpy as np
import pandas as pd
import seaborn as sns
import plotly.offline as py
from matplotlib import pyplot as plt

'''
state
State name abbreviation

numcol
Number of honey producing colonies

yieldpercol
Yield per colony (lbs)

totalprod
Total production (numcol*yieldpercol), (lbs)

stocks
Stocks held by producers on Dec 15 (lbs)

priceperlb
Average price per pound ($)

prodvalue
Value of production (totalprod*prodvalue), ($)

year
Year the data pertains to
'''

plt.style.use('fivethirtyeight')
py.init_notebook_mode()
state_code_to_name = {
    'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DC': 'District of Columbia',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'IA': 'Iowa',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'MA': 'Massachusetts',
    'MD': 'Maryland',
    'ME': 'Maine',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MO': 'Missouri',
    'MS': 'Mississippi',
    'MT': 'Montana',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'NE': 'Nebraska',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NV': 'Nevada',
    'NY': 'New York',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'PR': 'Puerto Rico',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VA': 'Virginia',
    'VT': 'Vermont',
    'WA': 'Washington',
    'WI': 'Wisconsin',
    'WV': 'West Virginia',
    'WY': 'Wyoming'
}

data = pd.read_csv('/Users/michaelwisniewski/desktop/PythonProjects/Kaggle/honeyproduction.csv' , sep=",").rename(columns={
    'state':'state_code',
    'numcol':'n_colony',
    'yieldpercol':'production_per_colony',
    'totalprod':'total_production',
    'stocks':'stock_held',
    'priceperlb':'price_per_lb',
    'prodvalue':'total_production_value'
})

data['consumption'] = data['total_production'] - data['stock_held']

data['state'] = data['state_code'].apply(lambda x: state_code_to_name[x])

inflation_rate = {
    1998: 1.454,
    1999: 1.423,
    2000: 1.376,
    2001: 1.339,
    2002: 1.317,
    2003: 1.288,
    2004: 1.255,
    2005: 1.214,
    2006: 1.176,
    2007: 1.143,
    2008: 1.101,
    2009: 1.105,
    2010: 1.087,
    2011: 1.054,
    2012: 1.032
}

monetized_features = ['price_per_lb', 'total_production_value']

for year in set(data['year']):
    for feature in monetized_features:
        data.loc[data['year']==year, feature] = inflation_rate[year]*data.loc[data['year']==year, feature]

data_by_year = data.groupby('year').mean()
data_by_year['production_per_colony_5e4'] = 50000*data_by_year['production_per_colony']
data_by_year[['total_production', 'production_per_colony_5e4', 'stock_held', 'total_production_value']].plot(ax=plt.subplots(figsize=(15,7))[1])
data_by_year[['price_per_lb']].plot(ax=plt.subplots(figsize=(15,3))[1])
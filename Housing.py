import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# %% codecell
df = pd.read_csv('housing.csv')
# print(df.describe())
# df.hist(bins=50, figsize=(20,20))
# sns.distplot(df)
df.columns

# %% codecell
from sklearn.model_selection import train_test_split

train_X, test_X = train_test_split(df, test_size=0.2, random_state=42)

# we want to start grouping up some of the data into smaller bins
df['income_cat'] = np.ceil(df['median_income'] / 1.5)
# print(df['income_cat'])
df['income_cat'] = df['income_cat'].where(df['income_cat'] <= 5, other=5.0)
sns.distplot(df['income_cat'], kde=False)
plt.figure(figsize=(8, 8))
plt.hist(df['income_cat'])

# %% codecell
from sklearn.model_selection import StratifiedShuffleSplit

# makes train and test dataset in a stratified manner accourding to the income_cat hist plots
# this ensures the train and test set samples represent a fair distribution of important attributes
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(df, df['income_cat']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

df['income_cat'].value_counts() / len(df)

for set in (strat_train_set, strat_test_set):
    set.drop('income_cat', axis=1, inplace=True)

# now we make a copy of the strat_train_set to make manipulations with
housing = strat_train_set.copy()

# %% codecell

# to visualize geographical data, then you can use a scatterplot
plt.figure(figsize=(12,12))
sns.scatterplot(x='longitude', y='latitude', data=housing, alpha=0.4, hue='median_house_value', size='population', sizes={10, 500})

# %% codecell

corr_matrix = housing.corr()
# print(corr_matrix['median_house_value'].sort_values())
# print(corr_matrix['median_house_value'].sort_values().iloc[0])

plt.figure(figsize=(12,12))
sns.catplot(x='index', y='median_house_value', data=corr_matrix.reset_index(), kind='bar')
plt.savefig('Correlation.png')














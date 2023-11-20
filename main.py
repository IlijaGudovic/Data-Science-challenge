import pandas as pd
#import numpy as np

dataset = pd.read_csv("jobfair_train.csv")
testing = pd.read_csv("jobfair_test.csv")

#Replacing string with index
def string_replacment(seris):

  visited_string = list(set(seris))
  for i in range(len(visited_string)):
    seris = seris.replace(visited_string[i], i)

  return seris

dataset['dynamic_payment_segment'] = string_replacment(dataset['dynamic_payment_segment'])
testing['dynamic_payment_segment'] = string_replacment(testing['dynamic_payment_segment'])

irrelevant_data = ['club_id', 'league_id', 'league_rank']

#Droping all non numeric values from table
dataset = dataset.select_dtypes(exclude=['object'])
testing = testing.select_dtypes(exclude=['object'])

#Replace nan with 0
dataset = dataset.fillna(0)
testing = testing.fillna(0)

del dataset['season']
del testing['season']

print(dataset.shape)
print(testing.shape)

#Methd correlation
def get_corr(column):
  corr_table = pd.DataFrame(columns=['league_rank', column])
  corr_table['league_rank'] = dataset['league_rank']
  corr_table[column] = dataset[column]
  corr = corr_table.corr()
  return corr.iloc[1, 0]

#Implementing norma
def normalize(array):
  return (array - array.min()) / (array.max() - array.min())

#Normalizing data
predict = [0] * testing.shape[0]

for column in testing:
    current = testing[column]
    dependence_factor = get_corr(column)
    predict = predict + normalize(current) * dependence_factor

testing['prediction'] = predict

#Sorting data
all_legues = list(set(testing['league_id']))

for league in all_legues:
  select = testing.loc[testing['league_id'] == league]
  select = select.sort_values(by=['prediction'], ascending=True)
  select = select.reset_index(drop=True)
  select.index += 1

  select['prediction'] = select.index.values.tolist()

  for index, row in select.iterrows():
    testing.prediction[testing.club_id==row.club_id] = row['prediction']

#Try
random_id = int(testing.sample(1).league_id)
print(testing[testing.league_id==random_id])

#Testing data
predict = [0] * dataset.shape[0]

for column in dataset:
    current = dataset[column]
    dependence_factor = get_corr(column)
    predict = predict + normalize(current) * dependence_factor

dataset['prediction'] = predict

all_legues = list(set(dataset['league_id']))

for league in all_legues:
  select = dataset.loc[dataset['league_id'] == league]
  select = select.sort_values(by=['prediction'], ascending=True)
  select = select.reset_index(drop=True)
  select.index += 1

  select['prediction'] = select.index.values.tolist()

  for index, row in select.iterrows():
    dataset.prediction[dataset.club_id==row.club_id] = row['prediction']

#Try
random_id = int(dataset.sample(1).league_id)
output=dataset[dataset.league_id==random_id]
print(output.league_rank, output.prediction)


#Evaulation -> Accuracy: 0.39192609465957984
from sklearn import metrics
print('Accuracy:', metrics.accuracy_score(dataset.prediction, dataset.league_rank))


#Exporting dataset
export = pd.DataFrame()
export['club_id'] = testing.club_id
export['predicted_league_rank'] = testing.prediction

export.to_csv('league_rank_predictions.csv', index = False)
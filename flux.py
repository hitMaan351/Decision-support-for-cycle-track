import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

flux_velo=pd.read_csv('/Users/thomasvicaire/Desktop/cours_cs/IA/challengeia/data/comptage-velo-donnees-compteurs.csv',sep=';')

## tests on dataset 
flux_velo2=flux_velo.head(100)
flux_velo2=flux_velo2[['sum_counts', 'name', 'mois_annee_comptage']]
#flux_velo2.groupby('name').agg({'sum_counts':'sum'})
df=pd.pivot_table(flux_velo,values=['sum_counts'], index=['name','mois_annee_comptage'],
                      aggfunc={'sum_counts':np.sum})

#df=df.reset_index()
df.name.unique()
df2=df[df['name']=='Voie Georges Pompidou']
df2
plt.plot(df2.mois_annee_comptage,df2.sum_counts)

## import of data from previous years 
df3=pd.read_csv('data/2018_comptage-velo-donnees-compteurs.csv', sep=';')
df4=pd.read_csv('data/2019_comptage-velo-donnees-compteurs-2.csv', sep=';')
df5=pd.read_csv('data/2020_comptage-velo-donnees-compteurs.csv', sep=';')
df6=pd.read_csv('data/2021_comptage-velo-donnees-compteurs.csv', sep=';')

#clean the format 
df3=df3[['Nom du site de comptage','Date et heure de comptage','Comptage horaire']].rename(columns={'Nom du site de comptage':'name','Date et heure de comptage':'mois_annee_comptage','Comptage horaire':'sum_counts'})
df4=df4[['Nom du site de comptage','Date et heure de comptage','Comptage horaire']].rename(columns={'Nom du site de comptage':'name','Date et heure de comptage':'mois_annee_comptage','Comptage horaire':'sum_counts'})
df5=df5[['Nom du site de comptage','Date et heure de comptage','Comptage horaire']].rename(columns={'Nom du site de comptage':'name','Date et heure de comptage':'mois_annee_comptage','Comptage horaire':'sum_counts'})
df6=df6[['Nom du site de comptage','Date et heure de comptage','Comptage horaire']].rename(columns={'Nom du site de comptage':'name','Date et heure de comptage':'mois_annee_comptage','Comptage horaire':'sum_counts'})

#change the format of dates
df3.mois_annee_comptage=df3.mois_annee_comptage.apply(lambda x: str(pd.to_datetime(x).year)+'-'+str(pd.to_datetime(x).month))
df4.mois_annee_comptage=df4.mois_annee_comptage.apply(lambda x: str(pd.to_datetime(x).year)+'-'+str(pd.to_datetime(x).month))
df5.mois_annee_comptage=df5.mois_annee_comptage.apply(lambda x: str(pd.to_datetime(x).year)+'-'+str(pd.to_datetime(x).month))
df6.mois_annee_comptage=df6.mois_annee_comptage.apply(lambda x: str(pd.to_datetime(x).year)+'-'+str(pd.to_datetime(x).month))

#aggregate count of bikes by month 
df3=pd.pivot_table(df3,values=['sum_counts'], index=['name','mois_annee_comptage'],
                      aggfunc={'sum_counts':np.sum}).reset_index()
df4=pd.pivot_table(df4,values=['sum_counts'], index=['name','mois_annee_comptage'],
                      aggfunc={'sum_counts':np.sum}).reset_index()
df5=pd.pivot_table(df5,values=['sum_counts'], index=['name','mois_annee_comptage'],
                      aggfunc={'sum_counts':np.sum}).reset_index()
df6=pd.pivot_table(df6,values=['sum_counts'], index=['name','mois_annee_comptage'],
                      aggfunc={'sum_counts':np.sum}).reset_index()


## other tests
df2=df3[df3['name']=='Voie Georges Pompidou']
df2
plt.plot(df2.mois_annee_comptage,df2['sum_counts'])

#creating the train data set
traindata=pd.concat([df,df3,df4,df5,df6])
traindata=traindata.drop(columns=['level_0','index'])

#sort for dataviz
traindata=traindata.sort_values(by='mois_annee_comptage')

#example of 'voie georges pompidou
plt.plot(traindata[traindata['name']=='Voie Georges Pompidou'].mois_annee_comptage,traindata[traindata['name']=='Voie Georges Pompidou'].sum_counts)


# save the dataset in a csv file
traindata.to_csv('data/traindata.csv')

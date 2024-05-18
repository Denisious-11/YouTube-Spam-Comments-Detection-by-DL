#importing necessary libraries
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd



df1=pd.read_csv('Dataset/Eminem.csv')
s_df1=df1[['CONTENT','CLASS']]
print(s_df1)
print(s_df1.columns)

df2=pd.read_csv('Dataset/KatyPerry.csv')
s_df2=df2[['CONTENT','CLASS']]


df3=pd.read_csv('Dataset/LMFAO.csv')
s_df3=df3[['CONTENT','CLASS']]


df4=pd.read_csv('Dataset/Psy.csv')
s_df4=df4[['CONTENT','CLASS']]


df5=pd.read_csv('Dataset/Shakira.csv')
s_df5=df5[['CONTENT','CLASS']]

final_df=s_df1.append(s_df2,ignore_index=True)
final_df=final_df.append(s_df3,ignore_index=True)
final_df=final_df.append(s_df4,ignore_index=True)
final_df=final_df.append(s_df5,ignore_index=True)


print(final_df)
print(final_df.columns)


final_df.to_csv("Final_Dataset/my_Dataset.csv",index=False)

#importing necessary libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM, Dense
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint 


def print_star():
    print('*'*50, '\n')

#reading dataset
df = pd.read_csv("Final_Dataset/my_Dataset.csv")
print("DATA LOADED\n")
print(df.head())
print(df.columns)

print_star()

#Preprocessing

#checking null values
print( df.isnull().sum())

# Seperating data and labels
data=df["CONTENT"]
labels=df["CLASS"]

print(labels.value_counts())
print_star()

def stemming(content):
	review = re.sub('[^a-zA-Z]',' ',content)
	review = review.split()
	review = [word for word in review if not word in stopwords.words('english')]
	review = ' '.join(review)
	return review

def cleantext(text):
    x=str(text).lower().replace('\\','').replace('_','')
    p_text=x.replace('<.*?>','') 
    p_text=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",p_text).split())
    p_text=p_text.replace('[^\w\s]','')
    return p_text

data=data.apply(lambda x:cleantext(x))
data= data.apply(stemming)
print(data)




#Train-Test Splitting
x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.3)
print("\nTraining set")
print(x_train.shape)
print(y_train.shape)
print("\nTesting set")
print(x_test.shape)
print(y_test.shape)

print_star()
print("\n")


#Calculate mean of all reviews length
def calc_mean():
    review_length=[]
    for review in x_train:
        review_length.append(len(review)) #calculate the length of all reviews & append to a list
    return int(np.ceil(np.mean(review_length))) #calculate mean of that list,rounding the mean,convert float to int
    
max_length=calc_mean()
print("Max_Length : ",max_length) #53


###################VECTORIZATION
#Feature extraction
token=Tokenizer(lower=False) #convert the review to tokens(words)
token.fit_on_texts(x_train)  #each word automatically indexed
x_train=token.texts_to_sequences(x_train)   #convert it into integers
x_test=token.texts_to_sequences(x_test)     #convert it into integers



###Padding(adding 0)/Truncating Reviews based on mean value
x_train=pad_sequences(x_train,maxlen=max_length,padding="post",truncating="post")#post->back of sentence
x_test=pad_sequences(x_test,maxlen=max_length,padding="post",truncating="post")

total_words=len(token.word_index)+1 #add 1 because of 0 padding

print("Encoded x_train\n",x_train,"\n")
print("Encoded x_test\n",x_test,"\n")





####################Building Model
EMBED_DIM=32
LSTM_OUT=64

#Model Architecture
model=Sequential()#this model takes sequences of data
model.add(Embedding(total_words,EMBED_DIM,input_length=max_length))#(size of vocabulary,size of output vector,input length)
model.add(LSTM(LSTM_OUT))
model.add(Dense(1,activation="sigmoid"))

#Compiling the model
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

#Printing model summary
print(model.summary())

print_star()

#saving the tokenizer
with open("Project_Saved_Models/tokenizer_lstm.pickle",'wb') as handle:
    pickle.dump(token,handle,protocol=pickle.HIGHEST_PROTOCOL)


###########################TRAINING

#saving the model(checkpoint)
checkpoint=ModelCheckpoint("Project_Saved_Models/lstm_model.h5",monitor="accuracy",save_best_only=True,verbose=1)#when training deep learning model,checkpoint is "WEIGHT OF THE MODEL"

#training 
history=model.fit(x_train,y_train,batch_size=128,epochs=40,callbacks=[checkpoint])



#plot accuracy and loss 
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()












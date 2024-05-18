#Importing necessary libraries
from tensorflow.keras.models import load_model #load  model
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle

english_stop_words=set(stopwords.words('english'))  #collecting stopwords

loaded_model=load_model("Project_Saved_Models/gru_model_94acc.h5")

comment=str(input('Enter Comment : '))

#################

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


pre1=cleantext(comment)
# print("PRE 1 : ",pre1)
pre2=stemming(pre1)
# print("PRE 2 : ",pre2)

# print(type(pre2))

pre2=[pre2]

max_length=54

#load tokenizer
with open("Project_Saved_Models/tokenizer_gru.pickle",'rb') as handle:
    token=pickle.load(handle)

tokenize_words=token.texts_to_sequences(pre2)
tokenize_words=pad_sequences(tokenize_words,maxlen=max_length,padding="post",truncating="post")
#print("tokenized:",tokenize_words)



result=loaded_model.predict(tokenize_words)
result=result[0][0]

if result>=0.5:
    print("[Danger] : SPAM Detected")
else:
    print("NO spam")

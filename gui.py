#Importing necessary libraries
from tkinter import *
from PIL import ImageTk, Image  
from tkinter import messagebox
from tensorflow.keras.models import load_model #load  model
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

english_stop_words=set(stopwords.words('english'))  #collecting stopwords

loaded_model=load_model("Project_Saved_Models/lstm_model_99acc.h5")

#load tokenizer
with open("Project_Saved_Models/tokenizer_lstm.pickle",'rb') as handle:
    token=pickle.load(handle)

max_length=53



def check(un_entry,password_entry1):
    username=un_entry.get()
    #print("username : ",username)
    password=password_entry1.get()
    #print("password : ",password)

    if(username=="" or password==""):
        messagebox.showwarning("warning","Please Fill Details")  
    elif(username=="admin" and password=="admin"):
        admin()
    else:
        messagebox.showwarning("warning","Invalid Credentials")  


def login():
    LoginPage = Frame(window)
    LoginPage.grid(row=0, column=0, sticky='nsew')
    LoginPage.tkraise()
    window.title('Youtube Spam Comment Detector')

    #login page
    de1 = Listbox(LoginPage, bg='#ffc14d', width=115, height=50, highlightthickness=0, borderwidth=0)
    de1.place(x=0, y=0)
    de2 = Listbox(LoginPage, bg= '#ffe6cc', width=115, height=50, highlightthickness=0, borderwidth=0)
    de2.place(x=606, y=0)

    de3 = Listbox(LoginPage, bg='#ffdb99', width=100, height=33, highlightthickness=0, borderwidth=0)
    de3.place(x=76, y=66)

    de4 = Listbox(LoginPage, bg='#f8f8f8', width=85, height=33, highlightthickness=0, borderwidth=0)
    de4.place(x=606, y=66)
    #  Username
    un_entry = Entry(de4, fg="#333333", font=("yu gothic ui semibold", 12), highlightthickness=2,
                        )
    un_entry.place(x=134, y=170, width=256, height=34)
    un_entry.config(highlightbackground="black", highlightcolor="black")
    un_label = Label(de4, text='• Username', fg="#89898b", bg='#f8f8f8', font=("yu gothic ui", 11, 'bold'))
    un_label.place(x=130, y=140)
    #  Password 
    password_entry1 = Entry(de4, fg="#333333", font=("yu gothic ui semibold", 12), show='*', highlightthickness=2,
                            )
    password_entry1.place(x=134, y=250, width=256, height=34)
    password_entry1.config(highlightbackground="black", highlightcolor="black")
    password_label = Label(de4, text='• Password', fg="#89898b", bg='#f8f8f8', font=("yu gothic ui", 11, 'bold'))
    password_label.place(x=130, y=220)

    # function for show and hide password
    def password_command():
        if password_entry1.cget('show') == '*':
            password_entry1.config(show='')
        else:
            password_entry1.config(show='*')

    # checkbutton 
    checkButton = Checkbutton(de4, bg='#f8f8f8', command=password_command, text='show password')
    checkButton.place(x=140, y=288)

    # Welcome Label 
    welcome_label = Label(de4, text='Welcome', font=('Arial', 20, 'bold'), bg='#f8f8f8')
    welcome_label.place(x=130, y=15)

    #top Login Button
    lob = Label(LoginPage, text='Login', font=("yu gothic ui bold", 12), bg='#f8f8f8', fg="#89898b",
                          borderwidth=0, activebackground='#1b87d2')
    lob.place(x=845, y=175)

    lol = Canvas(LoginPage, width=60, height=5, bg='black')
    lol.place(x=836, y=203)

    #  LOGIN  down button 
    loginBtn1 = Button(de4, fg='#f8f8f8', text='Login', bg='#1b87d2', font=("yu gothic ui bold", 15),
                       cursor='hand2', activebackground='#1b87d2',command=lambda:check(un_entry,password_entry1))
    loginBtn1.place(x=133, y=340, width=256, height=50)
    # User icon 
    u_icon = Image.open('images\\user.png')
    photo = ImageTk.PhotoImage(u_icon)
    Uicon_label = Label(de4, image=photo, bg='#f8f8f8')
    Uicon_label.image = photo
    Uicon_label.place(x=103, y=173)

    #  password icon 
    password_icon = Image.open('images\\key.png')
    photo = ImageTk.PhotoImage(password_icon)
    password_icon_label = Label(de4, image=photo, bg='#f8f8f8')
    password_icon_label.image = photo
    password_icon_label.place(x=103, y=253)
    #  picture icon 
    picture_icon = Image.open('images\\user-experience.png')
    photo = ImageTk.PhotoImage(picture_icon)
    picture_icon_label = Label(de4, image=photo, bg='#f8f8f8')
    picture_icon_label.image = photo
    picture_icon_label.place(x=280, y=6)

    #  Left Side Picture 
    side_image = Image.open('images\\youtube.png')
    side_image = side_image.resize((400,400))
    photo = ImageTk.PhotoImage(side_image)
    side_image_label = Label(de3, image=photo, bg='#ffdb99')
    side_image_label.image = photo
    side_image_label.place(x=70, y=65)

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

def predict(text1):
    comment=text1.get("1.0",'end')
    print("comment : ",comment)
    if comment=='' or comment=='\n':
        messagebox.showwarning("warning","Please Give an Input comment")
    else:
        list_box.insert(1, "Preprocessing")
        list_box.insert(2, "")
        list_box.insert(3, "Vectorization")
        list_box.insert(4, "")
        list_box.insert(5, "Perform Padding")
        list_box.insert(6, "")
        list_box.insert(7, "Load LSTM model")
        list_box.insert(8, "")
        list_box.insert(9, "Prediction")

        pre1=cleantext(comment)
        pre2=stemming(pre1)
        pre2=[pre2]

        tokenize_words=token.texts_to_sequences(pre2)
        tokenize_words=pad_sequences(tokenize_words,maxlen=max_length,padding="post",truncating="post")

        result=loaded_model.predict(tokenize_words)
        result=result[0][0]

        if result>=0.5:
            print("[Danger] : SPAM Detected")
            output="Spam Comment Detected"
        else:
            print("NO spam")
            output="[Safe] :No spam Comment"



        out_label.config(text=output)


    
def admin():
    Admin=Frame(window,bg="#5845d3")
    Admin.grid(row=0, column=0, sticky='nsew')
    Admin.tkraise()
    window.title('Youtube Spam Comment Detector')


    de2 = Listbox(Admin, bg='#fc766a', width=101, height=42, highlightthickness=0, borderwidth=0)
    de2.place(x=0, y=0)

    #light red
    de3 = Listbox(Admin, bg='#783937', width=115, height=30, highlightthickness=0, borderwidth=0)
    de3.place(x=606, y=0)
   

    #green
    de4 = Listbox(Admin, bg='#F1AC88', width=115, height=19, highlightthickness=0, borderwidth=0)
    de4.place(x=606, y=395)

    input_label = Label(de2, text='Input', font=('Arial', 24, 'bold'), bg='#FC766A')
    input_label.place(x=245, y=38)
    i1 = Canvas(de2, width=104, height=2, bg='#333333',highlightthickness=0)
    i1.place(x=230, y=82)

    # global text1
    text1=Text(de2,height=12,width=70)
    text1.place(x=20,y=132)


  
    process_label = Label(de3, text='Process', font=('Arial', 24, 'bold'), bg='#783937')
    process_label.place(x=245, y=38)
    lol = Canvas(de3, width=154, height=2, bg='#333333',highlightthickness=0)
    lol.place(x=230, y=82)

    global list_box
    list_box = Listbox(de3, height=14, width=40)
    list_box.place(x=180,y=102)


    result_label = Label(de4, text='Result', font=('Arial', 24, 'bold'), bg='#F1AC88')
    result_label.place(x=245, y=38)
    o1 = Canvas(de4, width=130, height=2, bg='#333333',highlightthickness=0)
    o1.place(x=230, y=82)

    global out_label
    out_label = Label(de4, text="", bg="#F1AC88", font="arial 18 bold")
    out_label.place(x=160, y=112)

  
    # Buttons
    p_b_image=Image.open('images\\pre_button.png')
    p_b_photo=ImageTk.PhotoImage(p_b_image)
    predict_Btn1 = Button(de2, image=p_b_photo, bg='#FC766A',
                       cursor='hand2',bd=0, activebackground='#6699ff',command=lambda:predict(text1))
    predict_Btn1.image=p_b_photo
    predict_Btn1.place(x=103, y=390)


    refresh_image=Image.open('images\\re_button.png')
    refresh_photo=ImageTk.PhotoImage(refresh_image)
    refresh_Btn1 = Button(de2, image=refresh_photo, bg='#FC766A',
                       cursor='hand2',bd=0, activebackground='#00cc66',command=lambda:admin())
    refresh_Btn1.image=refresh_photo
    refresh_Btn1.place(x=373, y=390)

    # predict_Btn1 = Button(de2, fg='#f8f8f8', text='Predict', bg='#3366ff', font=("yu gothic ui bold", 15),
    #                    cursor='hand2', activebackground='#668cff',command=lambda:predict(text1))
    # predict_Btn1.place(x=103, y=390, width=126, height=40)


    # Buttons
    # refresh_Btn1 = Button(de2, fg='#f8f8f8', text='Refresh', bg='#33cc33', font=("yu gothic ui bold", 15),
    #                    cursor='hand2', activebackground='#70db70',command=lambda:admin())
    # refresh_Btn1.place(x=373, y=390, width=126, height=40)
    
window = Tk()
window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)
# window.state('zoomed')
# window.resizable(0, 0)
window.geometry("1200x650")
window.maxsize(1200, 650)
window.minsize(1200, 650)
# Window Icon Photo
icon = PhotoImage(file='images\\pic-icon.png')
window.iconphoto(True, icon)
login()

window.mainloop()

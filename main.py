# %%
import pandas as pd
import numpy as np
import seaborn as sns # this library is based on the matplot lib but it is more advance in terms of graphs and data visuslisation and the contains some data sets 
import matplotlib.pyplot as plt # it is an open source library which used for data visuslization 
from sklearn.model_selection import train_test_split # this split the datat set into two parts ka training set-(used to train the machine learning model) and the testing set-( used to evaluate the models performance.) if dont provide the randomstate then the seed will be new everytime it runs...
from sklearn.metrics import accuracy_score ## used to calculate the total true positive and total true negavtive ratio which gives model effectivenesss  
from sklearn.metrics import classification_report  ## 
import re ## Python’s regular expressions module. Helps find and clean patterns in text (like links, emails, symbols).
import string ## Contains standard English characters, including string.punctuation to remove punctuation (!, ?, ., etc).
## TF-IDF helps you convert text into numbers, in a smart way.

## Part	Meaning of tf and idf
# TF (Term Frequency)	How often a word appears in one article. More = more important (locally).
# IDF (Inverse Document Frequency)	How rare the word is across all articles. Rare = more important (globally).




# %%
df_fake = pd.read_csv('Fake.csv')
df_true = pd.read_csv('True.csv')



# %%
df_fake.tail()

# %%
df_fake["class"]=0
df_true["class"]=1


# %%
df_fake.shape, df_true.shape

# %%
data_fake_manual_test= df_fake.tail(10)
for i in range(23480, 23470,-1):
    df_fake.drop([i],axis=0, inplace= True)

# print(data_fake_manual_test)



data_true_manual_test= df_true.tail(10)
for i in range(21416, 21406,-1):
    df_true.drop([i],axis=0, inplace= True)

# %%
df_fake.shape, df_true.shape

# %%
data_fake_manual_test['class']=0
data_true_manual_test['class']= 1


# %%
data_fake_manual_test.head()

# %%
data_true_manual_test.head()

# %%
data_mearge= pd.concat([df_fake, df_true], axis=0)
data_mearge.head()

# %%
df_fake.head()

# %%
df_true.tail()

# %%
data_mearge.tail()

# %%
data_mearge.shape


# %%
data_mearge.columns

# %%
data= data_mearge.drop(['title', 'subject', 'date'], axis=1)

# %%
data.shape
print(data)

# %%
# The sample() method in pandas is used to randomly shuffle or sample rows from a DataFrame.
# The frac attribute specifies the fraction of rows to return in the random sample.
# For example, frac=1 means return all rows in random order (i.e., shuffle the DataFrame).
# So, data=data.sample(frac=1) shuffles all the rows of the DataFrame 'data'.
data=data.sample(frac=1) 



# %%
data.head()

# %%
data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)


# %%
data.columns,data

# %%
def wordopt(text):
    text= text.lower()
    text=re.sub('\[.*?\]', '', text)
    text= re.sub("\\W"," ",text)
    text= re.sub("https?://\S+|www\.\S+"," ",text)
    text= re.sub("<.*?>+"," ",text)
    text= re.sub("[%s]"% re.escape(string.punctuation),'',text)
    text= re.sub("\n","",text)
    text= re.sub("\w*\d\w*","",text)
    return text


    # This function cleans and normalizes text data by:
    # 1. Converting text to lowercase.
    # 2. Removing text inside square brackets.
    # 3. Replacing non-word characters with spaces.
    # 4. Removing URLs.
    # 5. Removing HTML tags.
    # 6. Removing punctuation.
    # 7. Removing newline characters.
    # 8. Removing words containing numbers.
    # The cleaned text is then returned.

# %%
data['text']= data['text'].apply(wordopt)

# %%
x= data['text']
y= data['class']



# %%
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.25)


# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization= TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test= vectorization.transform(x_test)


# %%
from sklearn. linear_model import LogisticRegression
LR= LogisticRegression()
LR.fit(xv_train, y_train)

# %%
pred_lr= LR.predict(xv_test)

# %%
LR.score(xv_test, y_test)

# %%
print(classification_report(y_test, pred_lr))

# %%
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# %%
pred_dt= DT.predict(xv_test)

# %%
DT.score(xv_test, y_test)


# %%
print(classification_report(y_test, pred_dt))

# %%
from sklearn.ensemble import GradientBoostingClassifier 
GB= GradientBoostingClassifier(random_state=0)
GB.fit(xv_train, y_train)




# %%
predit_gb= GB.predict(xv_test)

# %%
GB.score(xv_test, y_test)


# %%
print(classification_report(y_test, predit_gb))

# %%
from sklearn.ensemble import RandomForestClassifier
RF= RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)


# %%
pred_rf = RF.predict(xv_test)


# %%
RF.score(xv_test, y_test)


# %%
print(classification_report(y_test, pred_rf))

# %%
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    return "Unknown"  # in case of unexpected value

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GB.predict(new_xv_test)
    pred_RFC = RF.predict(new_xv_test)
    
    print(
        "\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(
            output_label(pred_LR[0]),
            output_label(pred_DT[0]),
            output_label(pred_GBC[0]),
            output_label(pred_RFC[0])
        )
    )

# Example usage
news = str(input("Enter news text: "))
manual_testing(news)


# %%



# %%

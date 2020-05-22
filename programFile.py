#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset = pd.read_csv("bank.csv")


# In[3]:


dataset.info()


# In[4]:


dataset.head(3)


# In[5]:


import seaborn as sns
sns.set()


# In[6]:


sns.countplot(dataset["poutcome"])


# In[7]:


# as we can clearly see that poucome is mostly falling in unknown catergory so we can remove it.
#it will make our model little faster and might improve accurracy also
dataset.drop("poutcome", axis=1, inplace = True)


# In[8]:


dataset.head(2)


# In[9]:


#dataset["contact"]= dataset["contact"].str.replace("unknown", "NaN", case = False)


# In[10]:


sns.countplot(dataset["contact"])


# In[11]:


sns.heatmap(dataset.isnull(), cbar=False, yticklabels=False , cmap='viridis')
# we have to clear our data before using nerual network


# In[ ]:





# In[12]:


sns.countplot(dataset["housing"])


# In[13]:


sns.countplot(dataset['contact'], hue='loan', data=dataset)


# In[14]:


#Using feature engineering to Replacing the data which is not present

def func(cols):
    contact = cols[0]
    loan = cols[1]
    if pd.isnull(contact):
        if loan == "no":
            return "cellular"
        else:
            return "telephone"
    else:
        return contact
    


# In[15]:


dataset["contact"] = dataset[["contact", "loan"]].apply(func , axis=1)


# In[16]:


sns.heatmap(dataset.isnull(), cbar=False, yticklabels=False , cmap='viridis')


# In[17]:


dataset.info()


# In[18]:


dataset = dataset.dropna()
# we can drop the records which education we don't have as the NaN records are less and not every record is very important


# In[19]:


y = dataset["y"] #Predicted
y = pd.get_dummies(y, drop_first=True)


# In[20]:


sns.heatmap(dataset.isnull(), cbar=False, yticklabels=False , cmap='viridis')
#now our data is prefectly cleaned


# In[21]:


dataset = dataset[["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous"]]


# In[22]:


dataset.head(4)


# In[23]:


sns.countplot(dataset["campaign"])


# In[24]:


dataset


# In[25]:


#converting into dummy variables so that we don't get into dummy variable trap
#droping one columnd to remove collinearity
education = dataset["education"]
education = pd.get_dummies(education, drop_first=True)


# In[26]:


marital = dataset["marital"]
marital = pd.get_dummies(marital, drop_first=True)


# In[27]:


default = dataset["default"]
default = pd.get_dummies(default, drop_first=True)


# In[28]:


housing = dataset["housing"]
housing = pd.get_dummies(housing, drop_first=True)


# In[29]:


loan = dataset["loan"]
loan = pd.get_dummies(loan, drop_first=True)


# In[30]:


contact = dataset["contact"]
contact = pd.get_dummies(contact, drop_first=True)


# In[31]:


job = dataset["job"]
job = pd.get_dummies(job, drop_first=True)


# In[32]:


month = dataset["month"]
month = pd.get_dummies(month, drop_first=True)


# In[33]:


X = pd.concat([job, marital, education, default, dataset["balance"], housing, loan, contact, dataset["day"], month, dataset["duration"], dataset["campaign"], dataset["duration"]] ,  axis=1)


# In[34]:


X.columns


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


X.shape


# In[37]:


y.shape


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[39]:


from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.utils.np_utils import to_categorical


# In[40]:


model = Sequential()


# In[41]:


X.info()


# In[42]:


#Adding Hidden layers
model.add(Dense(units=100, input_dim=34, activation="relu"))
model.add(Dense(units=80, activation="relu"))
model.add(Dense(units=60, activation="relu"))
model.add(Dense(units=40, activation="relu"))
model.add(Dense(units=20, activation="relu"))
model.add(Dense(units=10, activation="relu"))
model.add(Dense(units=5, activation="relu"))
model.add(Dense(units=5, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))


# In[43]:


model.compile(optimizer=Adam(learning_rate=0.01), loss="binary_crossentropy", metrics=["accuracy"])


# In[44]:


model.fit(X_train, y_train, epochs=1, batch_size=1)


# In[45]:


test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy: %.2f%%', test_acc)


# In[ ]:





# In[49]:





# In[54]:


pred = model.evaluate(X_test, y_test)


# In[55]:


#this functin gives loss and accuracy


# In[57]:


print(pred[1]*100)


# In[68]:


acc = print("%.2f%%" % (test_acc*100))


# In[72]:


f = open("prediction.txt", "w")
f.write(str(pred[1]*100))
f.close()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
req = requests.get("https://www.worldometers.info/coronavirus/")


# In[3]:


from bs4 import BeautifulSoup
soup = BeautifulSoup(req.text , 'html.parser')


# In[4]:


tables = soup.find_all('table')
print(len(tables))
list=[]
My_list = [list.append(tables[i]['class']) for i in range(0,len(tables))]
list


# In[16]:


My_table = soup.find_all('table', {'class':['main_table_countries']})


# In[19]:


table_html = str(My_table)
table_html


# In[20]:


from IPython.core.display import HTML
HTML(table_html)


# In[113]:


row = My_table[0].find_all('tr')
len(row)


# In[131]:


Indexes = []*210
for i in range(8,210):
    Indexes.append(row[i].find('td').text)
    
for i in range(8,100):
    Indexes[i] = Indexes[i].replace("\n","")
    
Indexes


# In[132]:


Row = [0]*210
for i in range(8,210):
    Row[i] = row[i].find_all('td')
    
for i in range(8,210):
    for j in range(0,12):
        Row[i][j] = Row[i][j].text


# In[142]:


columns = My_table[0].find_all('th')
for i in range(0,len(columns)):
    columns[i] = columns[i].text
columns


# In[146]:


import pandas as pd
df = pd.DataFrame(Row[8:] , columns=columns , index = Indexes)
df


# In[149]:


df.to_excel('Covid.xlsx' , index = None, header=True)


# In[ ]:





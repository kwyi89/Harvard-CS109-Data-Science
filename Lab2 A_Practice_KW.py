
# coding: utf-8

# In[1]:

import cs109style
cs109style.customize_mpl()
cs109style.customize_css()

# special IPython command to prepare the notebook for matplotlib
get_ipython().magic(u'matplotlib inline')

# A standard Python library
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import requests
from pattern import web



# In[77]:

url = 'http://en.wikipedia.org/wiki/List_of_countries_by_past_and_future_population'
website_html = requests.get(url).text


# In[80]:

def get_population_html_tables(html):
    dom = web.Element(html)
    ### 0. step: look at html source!
    
    #### 1. step: get all tables
    #tbls = dom('table')
    #### 2. step: get all tables we care about
    tbls = dom.by_class('sortable wikitable')
    return tbls

tables = get_population_html_tables(website_html)
print "table length: %d" %len(tables)
for t in tables:
    print t.attributes


# In[24]:

def table_type(tbl):
    ### Extract the table type
    return tbl('th')[0].content   

# defaultdicts have a default value that is inserted when a new key is accessed
    
tables_by_type = defaultdict(list)
for tbl in tables:
    tables_by_type[table_type(tbl)].append(tbl)

print tables_by_type


# In[53]:

# Now extract population data for countries from all tables and store it in dictionary.
# Tables are divided by year of the time. So we must merge all tables into a long table.
def get_countries_population(tables):
    result = defaultdict(dict)
    
    for tbl in tables:
        headers = tbl('tr')
        first_header = headers[0]
        th_s = first_header('th')

        years = [int(val.content) for val in th_s if val.content.isnumeric()]
        year_indices = [idx for idx, val in enumerate(th_s) if val.content.isnumeric()]

        print years
        print year_indices
        
        # iterate over all tables, extract headings and actual data and combine data into single dict
        rows = tbl('tr')[1:]
        for row in rows:
            tds = row('td')
            country_name = tds[0]('a')[0].content
            population_by_year = [int(tds[colidx].content.replace(',','')) for colidx in year_indices]
            subdict = dict(zip(years, population_by_year))
            result[country_name].update(subdict)
        
    return result

result = get_countries_population(tables_by_type[u'Country or territory'])
print len(result), "Countries extracted"
print result[u'Canada']
print hash(1985), hash(2050)


# In[71]:

# create dataframe

df = pd.DataFrame.from_dict(result, orient='index')
# sort based on year
df.sort(axis=1,inplace=True)
print df


# In[72]:

subtable = df.iloc[0:2, 0:2]
print "subtable"
print subtable
print ""

column = df[1955]
print "column"
print column
print ""

row = df.ix[0] #row 0
print "row"
print row
print ""

rows = df.ix[:2] #rows 0,1
print "rows"
print rows
print ""

element = df.ix[0,1955] #element
print "element"
print element
print ""

# max along column
print "max"
print df[1950].max()
print ""

# axes
print "axes"
print df.axes
print ""

row = df.ix[0]
print "row info"
print row.name
print row.index
print ""

countries =  df.index
print "countries"
print countries
print ""

print "Austria"
print df.ix['Austria']


# In[73]:

plotCountries = ['Austria', 'Germany', 'United States', 'France']
    
for country in plotCountries:
    row = df.ix[country]
    plt.plot(row.index, row, label=row.name ) 
    
plt.ylim(ymin=0) # start y axis at 0

plt.xticks(rotation=70)
plt.legend(loc='best')
plt.xlabel("Year")
plt.ylabel("# people (million)")
plt.title("Population of countries")


# In[76]:

def plot_populous(df, year):
    # sort table depending on data value in year column
    df_by_year = df.sort(year, ascending=False)
    
    plt.figure()
    for i in range(5):  
        row = df_by_year.ix[i]
        plt.plot(row.index, row, label=row.name ) 
            
    plt.ylim(ymin=0)
    
    plt.xticks(rotation=70)
    plt.legend(loc='best')
    plt.xlabel("Year")
    plt.ylabel("# people (million)")
    plt.title("Most populous countries in %d" % year)

plot_populous(df, 2010)
plot_populous(df, 2050)


# In[ ]:




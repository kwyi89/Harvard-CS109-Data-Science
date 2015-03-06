
# coding: utf-8

## Lab 3: Exploratory Data Analysis for Classification using Pandas and Matplotlib

# ### Preliminary plotting stuff to get things going

# In[2]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:

get_ipython().system(u'~/anaconda/bin/pip install brewer2mpl')


# In[4]:

import brewer2mpl
from matplotlib import rcParams

#colorbrewer2 Dark2 qualitative color table
dark2_cmap = brewer2mpl.get_map('Dark2', 'Qualitative', 7)
dark2_colors = dark2_cmap.mpl_colors

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()


# In[5]:

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)


# ##1. The Olive Oils dataset

# Some of the following text is taken from the rggobi book (http://www.ggobi.org/book/). It is an excellent book on visualization and EDA for classification, and is available freely as a pdf from Hollis for those with a Harvard Id. Even though the book uses ggobi, a lot of the same analysis can be done in Mondrian or directly in Matplotlib/Pandas (albeit not interactively).

# <hr/>
# 
# "The Olive Oils data has eight explanatory variables (levels of fatty acids in the oils) and nine classes (areas of Italy). The goal of the analysis is to develop rules that reliably distinguish oils from the nine different areas. It is a problem of practical interest, because oil from some areas is more highly valued and unscrupulous suppliers sometimes make false claims about the origin of their oil. The content of the oils is a subject of study in its own right: Olive oil has high nutritional value, and some of its constituent fatty acids are considered to be more beneficial than others."
# 
# In addition, fatty acid contents vary with climate: this information is important in deciding which varieties to grow where.
# 
# 

# "Source: Forina, M., Armanino, C., Lanteri, S. & Tiscornia, E. (1983), Classification of Olive Oils from their Fatty Acid Composition, in Martens, H. and
# Russwurm Jr., H., eds, Food Research and Data Analysis, Applied Science
# Publishers, London, pp. 189–214. It was brought to our attention by Glover
# & Hopke (1992).
# 
# Number of rows: 572
# 
# Number of variables: 10
# 
# Description: This data consists of the percentage composition of fatty acids
# found in the lipid fraction of Italian olive oils. The data arises from a study
# to determine the authenticity of an olive oil."
# <hr/>

# In[6]:

from IPython.display import Image
Image(filename='Italy.png')


# In working with python I always remember: a python is a duck.
# 
# In working with pandas and matplotlib I dont always remember the syntax. A programmer is a good tool for converting Stack Overflow snippets into code. I almost always put what I am trying to do into google and go from there. 
# 
# That said, I found the following links very useful in understanding the Pandas mode, how things work.
# 
# * http://blog.yhathq.com/posts/R-and-pandas-and-what-ive-learned-about-each.html
# * http://www.bearrelroll.com/2013/05/python-pandas-tutorial/
# * http://manishamde.github.io/blog/2013/03/07/pandas-and-python-top-10/

# ##2. Loading and Cleaning

# Let's load the olive oil dataset into a pandas dataframe and have a look at the first 5 rows.

# In[7]:

df=pd.read_csv("./olive.csv")
df.head(5)


# Let's rename that ugly first column. 
# 
# *Hint*: A Google search for 'python pandas dataframe rename' points you at this <a href="http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.rename.html">documentation</a>.

# In[8]:

print df.columns
df.rename(columns={df.columns[0]:'areastring'},inplace=True)
df.columns


# Let's explore. Which unique regions and areas are contained in the dataset?

# In[9]:

print 'regions\t', df.region.unique()
print 'areas\t', df.area.unique()


# Let's create a *crosstab*ulation or contingency table of the factors.
# 
# *Hint*: A Google search for 'python pandas cross tabulation' points you at this <a href="http://pandas.pydata.org/pandas-docs/stable/reshaping.html#cross-tabulations">documentation</a>.
# 

# In[10]:

pd.crosstab(df.area,df.region)


# Do we need to clean the dataset before we can explore it a little more? Let's have a look.

# In[11]:

df.head()


# Let's get rid of the junk numbering in `df.areastring`. For single column Pandas Series we use `map`. Here's the <a href="http://pandas.pydata.org/pandas-docs/dev/generated/pandas.Series.map.html">documentation</a>.

# In[12]:

df.areastring=df.areastring.map(lambda x: x.split('.')[-1])
df.head()


# To access a specific subset of columns of a dataframe, we can use list indexing.

# In[13]:

df[['palmitic','oleic']].head()


# Notice that this returns a new object of type DataFrame.

# To access the series of entries of a single column, we could do the following.

# In[14]:

print df['palmitic']


# Notice the difference in the syntax. In the first example where we used list indexing we got a new DataFrame. In the second example we got a series corresponding to the column. 

# In[15]:

print "type of df[['palmitic']]:\t", type(df[['palmitic']]) 
print "type of df['palmitic']:\t\t", type(df['palmitic'])


# To access the series of values of a single column more conveniently, we can do this:

# In[16]:

df.palmitic


# ### YOUR TURN NOW (10 minutes)

# Get the unique areastrings of the dataframe `df`.

# In[17]:

#your code here
df.areastring.unique()


# Create a new dataframe `dfsub` by taking the list of acids and using pandas' `apply` function to divide all values by 100.
# 
# If you're not familiar with pandas' `apply` function, a Google search for 'python pandas dataframe apply' points you to the <a href="http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.apply.html">documentation</a>

# In[18]:

acidlist=['palmitic', 'palmitoleic', 'stearic', 'oleic', 'linoleic', 'linolenic', 'arachidic', 'eicosenoic']

#your code here
dfsub = df[acidlist].apply(lambda x: x/100.0)
dfsub.head()


# Notice that we can replace part of the dataframe by this new frame. Since we need the percentages, let's do this. The `Oleic` percentages should be in the 70s and 80s if you did this right.

# In[19]:

df[acidlist]=dfsub
df.head()


# ##2. Quick Intro to Matplotlib

# This is just a quick and dirty intro. Please read the excellent tutorial <a href="http://nbviewer.ipython.org/urls/raw.github.com/jrjohansson/scientific-python-lectures/master/Lecture-4-Matplotlib.ipynb">here</a>.

# In[20]:

fig=plt.figure()
plt.scatter(df.palmitic, df.linolenic)
axis = fig.gca() #get current axis
axis.set_title('linolenic vs palmitic')
axis.set_xlabel('palmitic')
axis.set_ylabel('linolenic')
#fig can be got with fig.gcf()


# In[21]:

plt.hist(df.palmitic, bins=20)


# There are many many more kinds of plots.

# A more object oriented approach sees us using the `subplots` function to set both figure and axis.

# In[22]:

fig, axes=plt.subplots(figsize=(10,10), nrows=2, ncols=2)
axes[0][0].plot(df.palmitic, df.linolenic)
axes[0][1].plot(df.palmitic, df.linolenic, '.')
axes[1][0].scatter(df.palmitic, df.linolenic)
axes[1][1].hist(df.palmitic)
fig.tight_layout()


# ###YOUR TURN NOW (10 minutes)
# 
# Make scatterplots of the acids in the list `yacids` against the acids in the list `xacids`. As the names suggest, plot the acids in `yacids` along the y axis and the acids in `xacids` along the x axis. Label the axes with the respective acid name. Set it up as a grid with 3 rows and 2 columns.

# In[23]:

xacids=['oleic','linolenic','eicosenoic']
yacids=['stearic','arachidic']

#your code here
fig, axes = plt.subplots(figsize=(10,10),nrows=len(xacids),ncols=len(yacids))
for i, xacid in enumerate(xacids):
    for j, yacid in enumerate(yacids):
        axes[i][j].scatter(df[xacid],df[yacid])
        axes[i][j].set_xlabel(xacid)
        axes[i][j].set_ylabel(yacid)
fig.tight_layout()


# ##3. Pandas Data Munging

# The first concept we deal with here is pandas `groupby`. The idea is to group a dataframe by the values of a particular factor variable. The documentation can be found <a href="http://pandas.pydata.org/pandas-docs/dev/groupby.html">here</a>.

# In[24]:

region_groupby = df.groupby('region')
print type(region_groupby)
region_groupby.head()


# The function `groupby` gives you a dictionary-like object, with the keys being the values of the factor, and the values being the corresponding subsets of the dataframe.

# In[25]:

for key, value in region_groupby:
    print "( key, type(value) ) = (", key, ",", type(value), ")"
    v=value

v.head()


# The `groupby` function also acts like an object that can be **mapped**. After the mapping is complete, the rows are put together (**reduced**) into a larger dataframe. For example, using the `describe` function. The documentation of the `describe` function can be found <a href="http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.describe.html">here</a>.

# In[26]:

dfrd=region_groupby.describe()
print type(dfrd)
dfrd.head(20)


# So, one may iterate through the groupby 'dictionary', get the pandas series from each sub-dataframe, and compute the standard deviation using the `std` function (find documentation of `std` <a href="http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.std.html">here</a>):

# In[27]:

vecs=[]
keys=[]
for key, value in region_groupby:
    k=key
    v=value.std()
print k, type(v), v


# Or one might let pandas take care of concatenating the series obtained by running `std` on each dataframe back into a dataframe for us. Notice that the output dataframe is automatically indexed by region for us!

# In[28]:

dfbystd=df.groupby('region').std()
dfbystd.head()


# Or one can use `aggregate` to pass an arbitrary function of to the sub-dataframe. The function is applied columnwise.

# In[29]:

dfbymean=region_groupby.aggregate(np.mean)
dfbymean.head()


# In[30]:

region_groupby.aggregate(lambda x: x.palmitic.sum()) #probably not what u had in mind :-)


# Or one can use `apply` to pass an arbitrary function to the sub-dataframe. This one takes the dataframe as argument.

# In[31]:

region_groupby.apply(lambda f: f.mean())


# In[32]:

region_groupby.apply(lambda f: f.palmitic.mean())


# Let's rename the columns in `dfbymean` and `dfbystd`.

# In[33]:

renamedict_std={k:k+"_std" for k in acidlist}
renamedict_mean={k:k+"_mean" for k in acidlist}
dfbystd.rename(inplace=True, columns=renamedict_std)
dfbymean.rename(inplace=True, columns=renamedict_mean) 
dfbystd.head()


# Pandas can do general merges. When we do that along an index, it's called a `join` (<a href="http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.join.html">documentation</a>). Here we make two sub-dataframes and join them on the common region index.

# In[34]:

dfpalmiticmean = dfbymean[['palmitic_mean']] 
dfpalmiticstd = dfbystd[['palmitic_std']] 

newdfbyregion=dfpalmiticmean.join(dfpalmiticstd)
newdfbyregion.head()


# ###YOUR TURN NOW (10 minutes)
# 
# Let's weight the palmitic acids content by a random weight. We'll first extract a subset of columns from `df` and then you will write a function to weigh the palmitic content by this random weight, delivering a weighted palmitic mean in the final dataframe.

# In[35]:

df.shape


# In[36]:

weights=np.random.uniform(size=df.shape[0])
smallerdf=df[['palmitic']]
otherdf=df[['region']]
otherdf['weight'] = weights
otherdf.head()


# Join `smallerdf` and `otherdf` on the index, into smallerdf

# In[37]:

#your code here
smallerdf=smallerdf.join(otherdf)
smallerdf.head()


# Now lets use these weights to compute a weighted average over the palmitic column.

# In[38]:

#your code here
def weightedave(x):
    return (x.palmitic*x.weight).sum()/x.weight.sum()
smallerdf.groupby('region').apply(weightedave)


# Finally aggregate the column percentages by summing them up over the regions.

# In[39]:

#your code here
def myfunc(x):
    return np.sum(x)

region_groupby.aggregate(myfunc)


# ## One Dimensional Exploratory Data Analysis (EDA) with Pandas

# In[40]:

rkeys=[1,2,3]
rvals=['South','Sardinia','North']
rmap={e[0]:e[1] for e in zip(rkeys,rvals)}
rmap


# Let's get a dataframe with just the acids.

# In[41]:

mdf2=df.groupby('region').aggregate(np.mean)
mdf2=mdf2[acidlist]
mdf2.head()


# Let's make a bar plot of the relative mean percentages of the acids. In pandas this is as simple as:

# In[42]:

ax=mdf2.plot(kind='barh', stacked=True)
ax.set_yticklabels(rvals)
ax.set_xlim([0,100])


# Well, that's kind of ugly. In the appendix we have some code showing how you can clean this plot up.

# The above graph get's proportions of all the acids in each region. We can ask the opposite question: for each acid, what's the distribution of regions?

# In[43]:

fig, axes=plt.subplots(figsize=(10,20), nrows=len(acidlist), ncols=1)
i=0
colors=[dark2_cmap.mpl_colormap(col) for col in [1.0,0.5,0.0]]
for ax in axes.flatten():
    acid=acidlist[i]
    seriesacid=df[acid]#get the Pandas series
    minmax=[seriesacid.min(), seriesacid.max()]
    counts=[]
    nbins=30
    histbinslist = np.linspace(minmax[0],  minmax[1], nbins)
    counts=-np.diff([seriesacid[seriesacid>x].count() for x in histbinslist]).min()
    for k,g in df.groupby('region'):
        style = {'histtype':'step', 'color':colors[k-1], 'alpha':1.0, 'bins':histbinslist, 'label':rmap[k]}
        ax.hist(g[acid],**style)
        ax.set_xlim(minmax)
        ax.set_title(acid)
        ax.grid(False)
    #construct legend
    ax.set_ylim([0, counts])
    ax.legend()
    i=i+1
fig.tight_layout()


# You can make a mask!

# In[44]:

mask=(df.eicosenoic < 0.05)
mask


# The first gives a count, the second is a shortcut to get a probability!

# In[45]:

np.sum(mask), np.mean(mask)


# Pandas supports conditional indexing: <a href="http://pandas.pydata.org/pandas-docs/dev/indexing.html#boolean-indexing">documentation</a>

# In[46]:

loweico=df[df.eicosenoic < 0.02]
pd.crosstab(loweico.area, loweico.region)


# ### YOUR TURN NOW (10 minutes)
# 
# You can see that oleic dominates, and doesn't let us see much about the other acids. Remove it and let's draw bar plots again.

# In[47]:

acidlistminusoleic=['palmitic', 'palmitoleic', 'stearic', 'linoleic', 'linolenic', 'arachidic', 'eicosenoic']
#your code here

ax = region_groupby.aggregate(np.mean)[acidlistminusoleic].plot(kind="barh", stacked=True)
ax.set_yticklabels(rvals)


# **Note that there are no eicosenoic acids in regions 2 and 3, which are Sardinia and the North respectively**

# ## Two-dimensional EDA with Pandas

# Let's write code to scatterplot acid against acid color coded by region. A more polished version is in the appendix

# In[48]:

# just do the boxplot without the marginals to split the north out
def make2d(df, scatterx, scattery, by="region", labeler={}):
    figure=plt.figure(figsize=(8,8))
    ax=plt.gca()
    cs=list(np.linspace(0,1,len(df.groupby(by))))
    xlimsd={}
    ylimsd={}
    xs={}
    ys={}
    cold={}
    for k,g in df.groupby(by):
        col=cs.pop()
        x=g[scatterx]
        y=g[scattery]
        xs[k]=x
        ys[k]=y
        c=dark2_cmap.mpl_colormap(col)
        cold[k]=c
        ax.scatter(x, y, c=c, label=labeler.get(k,k), s=40, alpha=0.4);
        xlimsd[k]=ax.get_xlim()
        ylimsd[k]=ax.get_ylim()
    xlims=[min([xlimsd[k][0] for k in xlimsd.keys()]), max([xlimsd[k][1] for k in xlimsd.keys()])]
    ylims=[min([ylimsd[k][0] for k in ylimsd.keys()]), max([ylimsd[k][1] for k in ylimsd.keys()])]
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xlabel(scatterx)
    ax.set_ylabel(scattery)
    ax.grid(False)
    return ax
a=make2d(df, "linoleic","arachidic", labeler=rmap)
a.legend(loc='upper right');


# **A nonlinear classifier could separate the north from Sardinia!**

# We use the really ugly trellis rplot interface in Pandas to do some hierarchical digging. We plot oleic against linoleic. **We can split Sardinia. We might be able to split East Liguria out but there could be significant misclassification.**

# In[49]:

import pandas.tools.rplot as rplot
dfcopy=df.copy()
dfcopy['region']=dfcopy['region'].map(rmap)
imap={e[0]:e[1] for e in zip (df.area.unique(), df.areastring.unique())}
#dfcopy['area']=dfcopy['area'].map(imap)
plot = rplot.RPlot(dfcopy, x='linoleic', y='oleic');
plot.add(rplot.TrellisGrid(['region', '.']))
plot.add(rplot.GeomPoint(size=40.0, alpha=0.3, colour=rplot.ScaleRandomColour('area')));

fig=plot.render()
print df.areastring.unique()


# ### YOUR TURN NOW (10 minutes)

# Plot palmitoleic against palimitic. **What can you separate?** Use the `dfcopy` dataframe.

# In[52]:

#your code here
plot = rplot.RPlot(dfcopy, x = 'palmitic', y='palmitoleic')
plot.add(rplot.TrellisGrid(['region','.']))
plot.add(rplot.GeomPoint(size=30, alpha=0.2, colour=rplot.ScaleRandomColour('area')))
fig=plot.render()


# ## Appendix: For you to try at home

# ### Marginal data: Rug plots and histograms 

# This code allows you to plot marginals using rug plots and histograms

# In[50]:

#adapted from https://github.com/roban/quarum/blob/master/margplot.py
from mpl_toolkits.axes_grid1 import make_axes_locatable
def setup_mhist(axes, figure):
    ax1=axes
    divider = make_axes_locatable(ax1)
    ax2 = divider.append_axes("top", 1.5, pad=0.0, sharex=ax1)
    ax3 = divider.append_axes("right", 1.5, pad=0.0, sharey=ax1)
    #xscale=yscale='log'
    #ax2.set_yscale(yscale)
    #ax3.set_xscale(xscale)
    #ax2.set_ylim([0,1])
    #ax3.set_xlim([0,5])
    ax2.grid(False)
    ax3.grid(False)
    ax2.grid(axis="y", color="white", linestyle='-', lw=1)
    ax3.grid(axis="x", color="white", linestyle='-', lw=1)
    remove_border(ax2, right=True, left=False)
    remove_border(ax3, right=False, left=True, bottom=False, top=True)
    figure.subplots_adjust(left=0.15, right=0.95)
    return [ax1,ax2,ax3]

#BUG: need to get appropriate min and max amongst the multiple marginal hists
#BUG: need to get highest frequency marked as label when we do this.
def make_mhist(axeslist, x, y, color='b', mms=8):
    ax1 = axeslist[0]
    ax2 = axeslist[1]
    ax3 = axeslist[2]
    #print list(ax2.get_yticklabels())
    for tl in (ax2.get_xticklabels() + ax2.get_yticklabels() +
               ax3.get_xticklabels() + ax3.get_yticklabels()):
        tl.set_visible(False)
    #for tl in ( ax2.get_xticklabels() + ax3.get_yticklabels()):
    #    tl.set_visible(False)
    histbinslist = [np.ceil(len(x)/20.), np.ceil(len(y)/20.)]
    histbinslist = copy.copy(histbinslist)
    #style = {'histtype':'stepfilled', 'color':color, 'alpha':0.6, 'normed':True, 'stacked':True}
    style = {'histtype':'stepfilled', 'color':color, 'alpha':0.4}
    nbins = histbinslist[0]
    x_range = [np.min(x), np.max(x)]
    histbinslist[0] = np.linspace(x_range[0],  x_range[1], nbins)

    ax2.hist(x, histbinslist[0], **style)

    nbins = histbinslist[1]
    y_range = [np.min(y), np.max(y)]
    histbinslist[1] = np.linspace(y_range[0], y_range[1], nbins)
    ax3.hist(y, histbinslist[1], orientation='horizontal', **style)


# In[51]:

import random
import copy
def scatter_by(df, scatterx, scattery, by=None, figure=None, axes=None, colorscale=dark2_cmap, labeler={}, mfunc=None, setupfunc=None, mms=8):
    cs=copy.deepcopy(colorscale.mpl_colors)
    if not figure:
        figure=plt.figure(figsize=(8,8))
    if not axes:
        axes=figure.gca()
    x=df[scatterx]
    y=df[scattery]
    if not by:
        col=random.choice(cs)
        axes.scatter(x, y, cmap=colorscale, c=col)
        if setupfunc:
            axeslist=setupfunc(axes, figure)
        else:
            axeslist=[axes]
        if mfunc:
            mfunc(axeslist,x,y,color=col, mms=mms)
    else:
        cs=list(np.linspace(0,1,len(df.groupby(by))))
        xlimsd={}
        ylimsd={}
        xs={}
        ys={}
        cold={}
        for k,g in df.groupby(by):
            col=cs.pop()
            x=g[scatterx]
            y=g[scattery]
            xs[k]=x
            ys[k]=y
            c=colorscale.mpl_colormap(col)
            cold[k]=c
            axes.scatter(x, y, c=c, label=labeler.get(k,k), s=40, alpha=0.3);
            xlimsd[k]=axes.get_xlim()
            ylimsd[k]=axes.get_ylim()
        xlims=[min([xlimsd[k][0] for k in xlimsd.keys()]), max([xlimsd[k][1] for k in xlimsd.keys()])]
        ylims=[min([ylimsd[k][0] for k in ylimsd.keys()]), max([ylimsd[k][1] for k in ylimsd.keys()])]
        axes.set_xlim(xlims)
        axes.set_ylim(ylims)
        if setupfunc:
            axeslist=setupfunc(axes, figure)
        else:
            axeslist=[axes]
        if mfunc:
            for k in xs.keys():
                mfunc(axeslist,xs[k],ys[k],color=cold[k], mms=mms);
    axes.set_xlabel(scatterx);
    axes.set_ylabel(scattery);
    
    return axes

def make_rug(axeslist, x, y, color='b', mms=8):
    axes=axeslist[0]
    zerosx1=np.zeros(len(x))
    zerosx2=np.zeros(len(x))
    xlims=axes.get_xlim()
    ylims=axes.get_ylim()
    zerosx1.fill(ylims[1])
    zerosx2.fill(xlims[1])
    axes.plot(x, zerosx1, marker='|', color=color, ms=mms)
    axes.plot(zerosx2, y, marker='_', color=color, ms=mms)
    axes.set_xlim(xlims)
    axes.set_ylim(ylims)
    return axes
    
#BUG: remove ticks and maybe even border on top and right


# In[52]:

ax=scatter_by(df, 'linoleic', 'eicosenoic', by='region', labeler=rmap, mfunc=make_rug, mms=20)
ax.grid(False)
ax.legend(loc='upper right');


# In[53]:

ax=scatter_by(df, 'linoleic', 'arachidic', by='region', labeler=rmap, setupfunc=setup_mhist, mfunc=make_mhist, mms=20)
ax.grid(False)
ax.legend(loc='upper right');


# In[54]:

ax=scatter_by(df, 'linoleic', 'eicosenoic', by='region', labeler=rmap, setupfunc=setup_mhist, mfunc=make_mhist, mms=20)
ax.grid(False)
ax.legend(loc='upper right');


# ### Probability distributions

# In[55]:

import scipy.stats as stats
mu=0.
sigma=1.
samples=np.random.normal(mu, sigma, 10000)
plt.hist(samples,bins=25, normed=True)
nd=stats.norm()
plt.hist(nd.rvs(size=10000), bins=25, alpha=0.5,normed=True)
x=np.linspace(-4.0,4.0,100)
plt.plot(x,nd.pdf(x))
plt.plot(x,nd.cdf(x))


# In[56]:

mean = [0,0]
cov = [[1,0],[0,5]] # diagonal covariance, points lie on x or y-axis
m=300
nrvs = np.random.multivariate_normal(mean,cov,(m,m))
duets=nrvs.reshape(m*m,2)
print duets[:,1]
normaldf=pd.DataFrame(dict(x=duets[:,0], y=duets[:,1]))
normaldf.head()
ax=scatter_by(normaldf, 'x', 'y',  figure=plt.figure(figsize=(8,10)),setupfunc=setup_mhist, mfunc=make_mhist, mms=20)
#ax.grid(False)


# In[57]:

H, xedges, yedges = np.histogram2d(normaldf.x, normaldf.y, bins=(50, 50), normed=True)
extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
plt.imshow(H, extent=extent, interpolation='nearest')
plt.colorbar()


# ### Miscellaneous Pandas Plotting tools: scatters, boxplots, and parallel co-ordinates

# In[58]:

from pandas.tools.plotting import scatter_matrix
scatter_matrix(df[['linoleic','arachidic','eicosenoic']], alpha=0.3, figsize=(10, 10), diagonal='kde');


# In[59]:

plt.figure(figsize=(24,5))
for key, group in df.groupby('region'):
    plt.subplot(int('13'+str(key)))
    group[acidlistminusoleic].boxplot(grid=False)
    ax=plt.gca()
    ax.set_title(rvals[key-1])
    remove_border(ax, left=False, bottom=False)
    ax.grid(axis="y", color="gray", linestyle=':', lw=1)


# In[60]:

from pandas.tools.plotting import parallel_coordinates
dfna=df[['region', 'palmitic', 'palmitoleic', 'stearic', 'oleic', 'linolenic', 'linoleic', 'arachidic', 'eicosenoic']]
dfna_norm = (dfna - dfna.mean()) / (dfna.max() - dfna.min())
dfna_norm['region']=df['region'].map(lambda x: rmap[x])
parallel_coordinates(dfna_norm, 'region', colors=[dark2_cmap.mpl_colormap(col) for col in [1.0,0.5,0.0]], alpha=0.05)


# ### Improving the pandas histograms

# In[61]:

ax2=mdf2.plot(kind='barh', stacked=True, color=dark2_colors, grid=False, legend=False)
remove_border(ax2, left=False, bottom=False)
ax2.grid(axis="x", color="white", linestyle='-', lw=1)
ax2.legend(loc='right', bbox_to_anchor=(1.3,0.5))
labels2=['South','Sardinia','North']
ax2.set_yticklabels(labels2)
ax2.set_ylabel('');
ax2.set_xlim(right=100.0);


# In[62]:

#your code here


# ### More details are only good at times

# It's hard to understand the graph below. A hierarchical approach as we have used is better.

# In[63]:

fig=plt.figure(figsize=(10,10))
ax=scatter_by(df, 'linoleic', 'arachidic', by='area', figure=fig, labeler=imap, setupfunc=setup_mhist, mfunc=make_mhist, mms=20)
ax.grid(False)
ax.legend(loc='right', bbox_to_anchor=(1.7,0.5));


# On the other hand, inspecting loads of scatter plots is not a bad idea!

# In[64]:

indices=np.tril_indices(8)
plts=[]
for i,j in zip(indices[0], indices[1]):
    if i!=j:
        plts.append((i,j))
print plts


# In[65]:

fig, axes = plt.subplots(nrows=14, ncols=2, figsize=(14,40));
k=0
af=axes.flatten()
for a in af:
    i,j=plts[k]
    a=scatter_by(df, acidlist[i], acidlist[j], by='region', axes=a, labeler=rmap, mfunc=make_rug, mms=20);
    a.grid(False);
    k=k+1
af[0].legend(loc='best');
fig.tight_layout();


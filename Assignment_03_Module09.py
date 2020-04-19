#!/usr/bin/env python
# coding: utf-8

# Assignment week 03 - Explore and cluster the neighborhoods in Toronto

# In[1]:


#install Beautiful Soup and requests for Web Scaping
get_ipython().system('pip install BeautifulSoup4')
get_ipython().system('pip install requests')


# In[2]:


#Get Data
# importing necessary libraries
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np


# In[3]:


# getting data from internet
source = requests.get("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M")
soup = BeautifulSoup(source.text, 'lxml')

#using soup object, iterate the .wikitable to get the data from the HTML page and store it into a list
data = []
columns = []
table = soup.find(class_='wikitable')
for index, tr in enumerate(table.find_all('tr')):
    section = []
    for td in tr.find_all(['th','td']):
        section.append(td.text.rstrip())
    
    #First row of data is the header
    if (index == 0):
        columns = section
    else:
        data.append(section)

#convert list into Pandas DataFrame
canada_df = pd.DataFrame(data = data,columns = columns)
canada_df.head()


# In[4]:


#Data Cleanup
#Remove Boroughs that are 'Not assigned'
canada_df = canada_df[canada_df['Borough'] != 'Not assigned']
canada_df.head()


# In[5]:


# More than one neighborhood can exist in one postal code area, combined these into one row with the neighborhoods separated with a comma
canada_df["Neighborhood"] = canada_df.groupby("Postal code")["Neighborhood"].transform(lambda neigh: ', '.join(neigh))

#remove duplicates
canada_df = canada_df.drop_duplicates()

#update index to be postcode if it isn't already
if(canada_df.index.name != 'Postal code'):
    canada_df = canada_df.set_index('Postal code')
    
canada_df.head()


# In[6]:


# If a cell has a borough but a Not assigned neighborhood, then the neighborhood will be the same as the borough
canada_df['Neighborhood'].replace("Not assigned", canada_df["Borough"],inplace=True)
canada_df.head()


# In[7]:


# More than one neighborhood can exist in one postal code area, combined these into one row with the neighborhoods separated with a comma
canada_df["Neighborhood"] = canada_df.groupby("Postal code")["Neighborhood"].transform(lambda neigh: ', '.join(neigh))

#remove duplicates
canada_df = canada_df.drop_duplicates()

#update index to be postcode if it isn't already
if(canada_df.index.name != 'Postal code'):
    canada_df = canada_df.set_index('Postal code')
    
canada_df.head()


# In[8]:


# If a cell has a borough but a Not assigned neighborhood, then the neighborhood will be the same as the borough
canada_df['Neighborhood'].replace("Not assigned", canada_df["Borough"],inplace=True)
canada_df.head()


# In[9]:



canada_df.shape


# In[10]:


#Add Geospatial Data
#Get data lat/long data from csv
#lat_long_coord_df = pd.read_csv("Geospatial_Coordinates.csv")
lat_long_coord_df = pd.read_csv("http://cocl.us/Geospatial_data")

#rename columns and set the index to be Postcode
lat_long_coord_df.columns = ["Postal code", "Latitude", "Longitude"]
if(lat_long_coord_df.index.name != 'Postal code'):
    lat_long_coord_df = lat_long_coord_df.set_index('Postal code')
    
lat_long_coord_df.head()


# In[11]:


canada_df = canada_df.join(lat_long_coord_df)
canada_df.head(11)


# In[21]:


#Explore and Cluster Data
import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#pip install geopy
#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')


# In[22]:


#Get Toronto Data
#Filter Canada data to only use boroughs in Toronto
toronto_df = canada_df[canada_df['Borough'].str.contains('Toronto')]
toronto_df.head()


# In[15]:


#Generate Map
#Show an initial map of the neighborhoods in Toronto
# create map of Toronto using first entries latitude and longitude values
map_toronto = folium.Map(location=[toronto_df["Latitude"][0], toronto_df["Longitude"][0]], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(toronto_df['Latitude'], toronto_df['Longitude'], toronto_df['Borough'], toronto_df['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# In[23]:


#Use Four Square API to Explore
CLIENT_ID = 'GMQUUFMNMYGHULUNQEGZYZVGIWFZIL3V1ZEE5O2TGB3JZZRC' # your Foursquare ID
CLIENT_SECRET = 'KE1ATQ1DNLLRMDL4XNTC1BROVJEZAWRZKOL4NPL2OJZ45GYQ' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[24]:


#Explore First Neighborhood
# Get data of first neighborhood
neighborhood_latitude = toronto_df['Latitude'][0] # neighborhood latitude value
neighborhood_longitude = toronto_df['Longitude'][0] # neighborhood longitude value

neighborhood_name = toronto_df['Neighborhood'][0] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# In[25]:


# Setup API URL to explore venues near by
LIMIT = 100
radius = 500
url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, neighborhood_latitude, neighborhood_longitude, VERSION, radius, LIMIT)
neighborhood_json = requests.get(url).json()

# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
    
venues = neighborhood_json['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[26]:


#Generalize Venue Data Collection
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[27]:


#Get Venue Data from Foursquare for all neighborhoods
#Get all Tor
toronto_venues_df = getNearbyVenues(names=toronto_df['Neighborhood'],
                                   latitudes=toronto_df['Latitude'],
                                   longitudes=toronto_df['Longitude']
                                  )


# In[28]:


print(toronto_venues_df.shape)
toronto_venues_df.head()


# In[29]:


toronto_venues_df.groupby('Neighborhood').count()


# In[30]:



print('There are {} uniques categories.'.format(len(toronto_venues_df['Venue Category'].unique())))


# In[31]:


#Analyze Neighborhoods
# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues_df[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = toronto_venues_df['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# In[32]:


toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped


# In[33]:


num_top_venues = 5

for hood in toronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[34]:



#method to sort venues
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[35]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[36]:


#Cluster Neighborhoods
# set number of clusters
kclusters = int(len(toronto_df["Neighborhood"].unique()) / 4)
toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=1).fit(toronto_grouped_clustering)

# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = toronto_df.copy()
toronto_merged.rename(columns={'Neighborhood':'Neighborhood'}, inplace=True)
# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head() # check the last columns!


# In[37]:


# create map
map_clusters = folium.Map(location=[toronto_df["Latitude"][0], toronto_df["Longitude"][0]], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[ ]:





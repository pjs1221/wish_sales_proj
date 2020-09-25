# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 03:05:00 2020

@author: shinp
"""



import pandas as pd
import datetime
import math

df = pd.read_csv('Data/summer-products-with-rating-and-performance_2020-08.csv')



#Add column for number of total listings from the same merchant
df['num_listings'] = df['merchant_title'].apply(lambda x: df['merchant_title'].value_counts()[str(x)])

#Change nan in has_urgency_banner to 0
df['has_urgency_banner'] = df['has_urgency_banner'].apply(lambda x: 1 if x == 1 else 0)

#Create a list of the tags
df['tag_list'] = df['tags'].apply(lambda x: x.split(','))

#Change nan rating counts to 0
df['rating_five_count'] = df['rating_five_count'].apply(lambda x: 0 if math.isnan(x) else x)
df['rating_four_count'] = df['rating_four_count'].apply(lambda x: 0 if math.isnan(x) else x)
df['rating_three_count'] = df['rating_four_count'].apply(lambda x: 0 if math.isnan(x) else x)
df['rating_two_count'] = df['rating_two_count'].apply(lambda x: 0 if math.isnan(x) else x)
df['rating_one_count'] = df['rating_one_count'].apply(lambda x: 0 if math.isnan(x) else x)

#Change nan product colors to 'no_color'
df['product_color'] = df['product_color'].apply(lambda x: 'no_color' if not isinstance(x, str) else x)

#Change nan sizes to 'no_size'
df['product_variation_size_id'] = df['product_variation_size_id'].apply(lambda x: 'no_size' if not isinstance(x, str) else x)

#Change nan origin countries to 'no_origin_country'
df['origin_country'] = df['origin_country'].apply(lambda x: 'no_origin_country' if not isinstance(x, str) else x)


# Use imagga api to analyze whether or not image has person
response_list = []
import requests
def contains_person(img_url):
    
    url = "https://api.imagga.com/v2/categories/personal_photos"
    
    querystring = {"image_url":img_url}
    #Authorization key given by Imagga
    auth_key = "" 
    
    headers = {
        'accept': "application/json",
        'authorization': auth_key
        }
    
    response = requests.request("GET", url, headers=headers, params=querystring)
    txt = response.text
    response_list.append(txt)
    if('people' in txt):
        return 1
    else:
        return 0

df['image_contains_person'] = df['product_picture'].apply(lambda x: contains_person(x))

#Remove redundant/unneeded variables
del df['shipping_option_name']
del df['title']
del df['currency_buyer']
del df['urgency_text']
del df['merchant_name']
del df['merchant_profile_picture']
del df['merchant_id']
del df['merchant_info_subtitle']
del df['product_url']
del df['product_picture']
del df['product_id']
del df['theme']
del df['crawl_month']

#Output cleaned data to new csv
df_out = df

#Output cleaned data to new csv
df_out.to_csv('wish_sales_data_cleaned.csv',index = False)



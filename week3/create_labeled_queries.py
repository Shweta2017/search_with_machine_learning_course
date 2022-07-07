import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import unicodedata
from nltk.stem.snowball import SnowballStemmer

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
snow_stemmer = SnowballStemmer(language='english')
def transform_name(product_name):
    product_name = ''.join(c for c in unicodedata.normalize('NFD', product_name) if unicodedata.category(c) != 'Mn')
    product_name = ''.join(ch if ch.isalnum() else ' ' for ch in product_name)
    product_name_tokens = product_name.split()
    product_name_tokens = [snow_stemmer.stem(x.lower()) for x in product_name_tokens]
    product_name = ' '.join(product_name_tokens)
    return product_name

df['query'] = df['query'].apply(lambda x: transform_name(x))
# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
parent_list = [root_category_id]
level = 1
parents_df['level'] = 0
while True:
    affected_df = parents_df[parents_df.parent.isin(parent_list)]
    if affected_df.empty:
        break
    parents_df['level'] = parents_df.apply(lambda x: level if x['parent'] in parent_list else x['level'], axis=1)
    level = level + 1
    parent_list = affected_df.category.to_list()


while True:
    category_df = df.groupby('category')['query'].count().reset_index(name='frequency')
    categories_below_threshold = category_df[category_df.frequency < min_queries]
    if categories_below_threshold.empty:
        print("1. Distinct category", category_df.shape[0])
        break
    categories_below_threshold = pd.merge(categories_below_threshold, parents_df, on='category')
    max_level = categories_below_threshold.level.max()
    if max_level == 1:
        print("2. Distinct category", category_df.shape[0])
        break
    pruning_candidate_list = categories_below_threshold[categories_below_threshold.level == max_level].category.to_list()
    df['category'] = df['category'].apply(
        lambda x: parents_df.loc[parents_df.category == x]['parent'].iloc[0] if x in pruning_candidate_list else x)

# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)

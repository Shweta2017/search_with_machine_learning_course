import pandas as pd
import unicodedata
import csv

def filter_label_below_threshold(threshold=500):
    df = pd.read_csv('/workspace/datasets/fasttext/labeled_products.txt', sep='\t', 
        header=None, names=["raw"])
    df['label'] = df['raw'].apply(lambda x: x.split()[0])
    df['title'] = df['raw'].apply(lambda x: ' '.join(x.split()[1:]))
    df.drop(columns=['raw'], inplace=True)
    df_label_count = df.groupby('label').agg({'title': 'count'}).reset_index()
    df_label_count.rename(columns={'title': 'title_count'}, inplace=True)
    df = pd.merge(df, df_label_count, on='label')
    df = df[df.title_count >= threshold]
    df['raw'] = df.apply(lambda x: x['label'] + ' ' + x['title'], axis=1)
    df = df[['raw']]
    df = df.sample(frac=1)
    df.to_csv('/workspace/datasets/fasttext/shuffled_labeled_products.txt', header=False, index=False, 
        sep='\t', quoting = csv.QUOTE_NONE)


def create_synonym_training_data():
    df = pd.read_csv('/workspace/datasets/fasttext/output.fasttext', sep='\t', 
        header=None, names=["raw"])
    df['title'] = df['raw'].apply(lambda x: ' '.join(x.split()[1:]))
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()

    def lemmatize(row):
        word_list = row["title"].split()
        word_list = [wnl.lemmatize(z) for z in word_list]
        row['title'] = ' '.join(word_list)
        return row
    
    df = df.apply(lemmatize, axis=1)
    #df['title'] = df['title'].apply(lambda s: ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))
    df.drop(columns=['raw'], inplace=True)
    df = df.sample(frac=1)
    print(df.head())
    df.to_csv('/workspace/datasets/fasttext/titles.txt', header=False, index=False, 
        sep='\t', quoting = csv.QUOTE_NONE)


def get_synonym_candidate():
    df = pd.read_csv('/workspace/datasets/fasttext/titles.txt', sep='\t', header=None, names=["title"])
    word_freq_dict = {}
    for i, row in df.iterrows():
        row = dict(row)
        tokens = row['title'].split()
        for token in tokens:
            if len(token) < 4:
                continue
            if token in word_freq_dict:
                word_freq_dict[token] += 1
            else:
                word_freq_dict[token] = 1
    freq_list = []
    cnt = 0
    threshold = 1000
    for k,v in word_freq_dict.items():
        freq_list.append({'token': k, 'freq': v})
    freq_df = pd.DataFrame(freq_list)

    freq_df = freq_df.sort_values(by=['freq'], ascending=False)
    freq_df = freq_df.head(threshold)
    freq_df = freq_df[['token']]
    freq_df.to_csv('/workspace/datasets/fasttext/top_words.txt', header=False, index=False, 
        sep='\t', quoting = csv.QUOTE_NONE)


def generate_synonym_file():
    import fasttext
    model = fasttext.load_model("/workspace/datasets/fasttext/title_model_lemma.bin")
    threshold = 0.75
    df = pd.read_csv('/workspace/datasets/fasttext/top_words.txt', sep='\t', header=None, names=["token"])
    tokens = df['token'].tolist()
    synonyms_dict = []
    for token in tokens:
        synonyms_data = model.get_nearest_neighbors(token)
        nn = [token]
        for entry in synonyms_data:
            if entry[0] > threshold:
                nn.append(entry[1])
        if len(nn) > 1:
            synonyms_dict.append({"synonym": ', '.join(nn)})
    synonym_df = pd.DataFrame(synonyms_dict)
    print(synonym_df.head())
    synonym_df.to_csv('/workspace/datasets/fasttext/synonyms.csv', header=False, index=False, 
        sep='\t', quoting = csv.QUOTE_NONE)
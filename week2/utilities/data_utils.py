import pandas as pd

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
    df = df[['label', 'title']]
    print(df.head())
    df.to_csv('/workspace/datasets/fasttext/pruned_labeled_products.txt', header=False, index=False, sep=' ')

filter_label_below_threshold()


# LDA-model

#import package
```
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis  
import pyLDAvis.sklearn 
```

#import stopwords
```
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
stopwords = stopwordslist('stopwords-zh.txt')
```

#clean the content
```
data_ill['content_cutted'] = data_ill['概况'].apply(lambda x: jieba.cut(x,cut_all=False))
data_ill['content_cutted'] = data_ill['content_cutted'].apply(lambda x: ' '.join(item for item in x if item not in stopwords and item != ' '))
```

#设置关键词的数量
```
n_features = 20
```

```
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',  
                                max_features=n_features,  
                                stop_words='english',  
                                max_df = 0.5,  
                                min_df = 10)  
tf = tf_vectorizer.fit_transform(data_ill.content_cutted)

lda = LatentDirichletAllocation(n_components=4, max_iter=50,  
                                learning_method='online',  
                                learning_offset=50.,  
                                random_state=0)

lda.fit(tf)

```
#设置显示主题的内容

```
def print_top_words(model, feature_names, n_top_words):  
    for topic_idx, topic in enumerate(model.components_):  
        print("Topic #%d:" % topic_idx)  
        print(" ".join([feature_names[i]  
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))  
    print()
```
#显示内容
```
n_top_words = 20
tf_feature_names = tf_vectorizer.get_feature_names()  
print_top_words(lda, tf_feature_names, n_top_words)
```
#显示互动性内容
```
 
pyLDAvis.enable_notebook()  
pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)

```





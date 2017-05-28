# Predict categories from text with pre-trainder word embeddings on dmoz dataset

## The plan

1. ~~get the data from [Harvard](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OMV93V)~~
2. ~~use a crawler and an instance in the cloud to crawl 3+ million web page and store the HTML they return~~
3. ~~go through the categories tree and select a subset of categories to predict~~
4. ~~get word embeddings from [Standford](http://nlp.stanford.edu/data/glove.6B.zip)~~
5. ~~extact the text from the data and run Spark MLlib TF-IDF on the corpus~~
6. ~~For web-pages in english extend [this code](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) to train a NN predicting all the categories from 3 ~~
7. ~~Load the model in a JVM server and predict categories~~ 
8. Build a webservice that given a text will return predictions for categories


# Machine Learning Notebooks
I interned at DataFlair for a period of nearly 40 days. I did a total of 10 projects in Computer Vision, NLP and in Machine Learning.

This repository contains the notebooks along with the links to the blogs explaining each of the projects I did. 

### [Brain Tumor Detection](https://github.com/Jahnavi-Majji/ML-Notebooks/blob/main/brain_tumor_detection.ipynb)
* Dataset can be downloaded from [here](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
* Utilized Transfer Learning to build the classifier
* Achieved **96.5%** accuracy on the test set
* [Link to the article] (https://data-flair.training/blogs/brain-tumor-classification-machine-learning/)

### [Credit Card Fraud Detection](https://github.com/Jahnavi-Majji/ML-Notebooks/blob/main/credit_card_fraud.ipynb)
* Dataset can be downloaded from [here](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* Data suffers from a serious problem of **class-imbalance**. Only **0.17%** of the transactions were fraud.
* If we don't address the above problem, the model will easily get high accuracy and will not perform well in the real world.
* Used SMOTE to perform Oversampling.
* Applied Random Forest and Decision Trees algorithms and visualized the results.
* It turns out that the Random Forest algorithm with Oversampling performs even better than the other two models.
* Achieved more than **99%** on the accuracy, precision, recall and F1-score metrics
* [Link to the article] (https://data-flair.training/blogs/credit-card-fraud-detection-python-machine-learning/)


### [Movie Recommendation System](https://github.com/Jahnavi-Majji/ML-Notebooks/blob/main/Movie%20Recommendation.ipynb)
* Dataset can be downloaded from [here](https://www.kaggle.com/tmdb/tmdb-movie-metadata)
* Built a *basic* recommendation system using the IMDb weighted average score. This is called **Demographic Filtering**
* Demographic Filtering only gives the top results of all time. The results are not personalized. 
* Built a **Content Based Recommendation System** that recommends movies which are nearer to a movie's plot.
* Used **TfidfVectorizer** to represent a matrix where each column represents a word in the overview vocabulary (all the words that appear in at least one document) and each row represents a movie
* Used **cosine similarity** scores to calculate the similarity of plots (documents).
* The above system does well but it can be further improved by using the metadata of the movies.
* Used the cast, crew and keywords features to improve the model's recommendations.
* [Link to the article] (https://techvidvan.com/tutorials/movie-recommendation-system-python-machine-learning/)


### [Fake News Detection](https://github.com/Jahnavi-Majji/ML-Notebooks/blob/main/Fake%20News%20Detection.ipynb)
* Dataset can be downloaded from [here](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
* The text is preprocessed by removing all the letters except the alphabets and further applying Lemmatization.
* Applied MultiNomial Naive Bayes algorithm to train the model on the corpus.
* Achieved more than **95%** accuracy on both the training and test sets.
* [Link to the article] (https://projectgurukul.org/fake-news-detection-project-python-machine-learning/)

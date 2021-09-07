# Machine Learning Notebooks
I interned at [DataFlair](https://data-flair.training/) for a period of nearly 60 days. I did a total of 10 projects in Computer Vision, NLP and in Machine Learning.

This repository contains all of the notebooks along with the links to the blogs explaining each of the projects I did. All of the projects were done using TensorFlow and Keras API. Some of them were not published due to my period as an intern was terminated. So, the links will head to a google document explaining my project.

The other two notebooks are [Image captioning](https://github.com/jahn-chan/Image-Caption-Generator) and [Handwritten Text Generation](https://github.com/jahn-chan/Handwritten-Characters-Generation-using-GANs/).

### [Brain Tumor Detection](https://github.com/jahn-chan/ML-Notebooks/blob/main/brain_tumor_detection.ipynb)
* Dataset can be downloaded from [here](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
* Utilized Transfer Learning to build the classifier
* Achieved **96.5%** accuracy on the test set
* [Link to the article](https://data-flair.training/blogs/brain-tumor-classification-machine-learning/)

### [Credit Card Fraud Detection](https://github.com/jahn-chan/ML-Notebooks/blob/main/credit_card_fraud.ipynb)
* Dataset can be downloaded from [here](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* Data suffers from a serious problem of **class-imbalance**. Only **0.17%** of the transactions were fraud.
* If we don't address the above problem, the model will easily get high accuracy and will not perform well in the real world.
* Used SMOTE to perform Oversampling.
* Applied Random Forest and Decision Trees algorithms and visualized the results.
* It turns out that the Random Forest algorithm with Oversampling performs even better than the other two models.
* Achieved more than **99%** on the accuracy, precision, recall and F1-score metrics
* [Link to the article](https://data-flair.training/blogs/credit-card-fraud-detection-python-machine-learning/)


### [Movie Recommendation System](https://github.com/jahn-chan/ML-Notebooks/blob/main/Movie%20Recommendation.ipynb)
* Dataset can be downloaded from [here](https://www.kaggle.com/tmdb/tmdb-movie-metadata)
* Built a *basic* recommendation system using the IMDb weighted average score. This is called **Demographic Filtering**
* Demographic Filtering only gives the top results of all time. The results are not personalized. 
* Built a **Content Based Recommendation System** that recommends movies which are nearer to a movie's plot.
* Used **TfidfVectorizer** to represent a matrix where each column represents a word in the overview vocabulary (all the words that appear in at least one document) and each row represents a movie
* Used **cosine similarity** scores to calculate the similarity of plots (documents).
* The above system does well but it can be further improved by using the metadata of the movies.
* Used the cast, crew and keywords features to improve the model's recommendations.
* [Link to the article](https://techvidvan.com/tutorials/movie-recommendation-system-python-machine-learning/)


### [Fake News Detection](https://github.com/jahn-chan/ML-Notebooks/blob/main/Fake%20News%20Detection.ipynb)
* Dataset can be downloaded from [here](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
* The text is preprocessed by removing all the letters except the alphabets and further applying Lemmatization.
* Applied MultiNomial Naive Bayes algorithm to train the model on the corpus.
* Achieved more than **95%** accuracy on both the training and test sets.
* [Link to the article](https://projectgurukul.org/fake-news-detection-project-python-machine-learning/)

### [Emoji Prediction from the Text](https://github.com/jahn-chan/ML-Notebooks/blob/main/Emoji%20Prediction.py)
* This is really a cool project I did over the course of my internship.
* The text is preprocessed and then the word embedding matrix is computed using the pre-trained glove vector 6B 50D.
* Used LSTMs with Dropout mechanism to improve the accuracy. The architecture of the model consisted of 2 LSTM layers followed by a Dense layer.
* Achieved only **62%** accuracy on the test sets. There is a room for improvement.
* [Link to the article](https://data-flair.training/blogs/emoji-prediction-deep-learning/)

### [Sentiment Analysis](https://github.com/Jahnavi-Majji/jahn-chan/blob/main/Sentiment%20Analysis.py)
* I've always thought this project would be so easy as this is considered as the "Hello World" in the NLP projects.
* But, I learned a lot while doing this project and this introduced me to so many techniques.
* Dataset can be downloaded from [here](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
* The text is preprocessed by applying Tokenizer() on the corpus, then assigned each unique word a unique number and replacing each word with its assigned unqiue number. 
* The text is padded with zeroes to make the dimesions uniform for each text.
* Used LSTMs with Dropout mechanism to improve the accuracy to train the model on the dataset.
* Obtained an accuracy of **94%** on the validation set.
* [Link to the article](https://techvidvan.com/tutorials/python-sentiment-analysis/)

### [Pedestrian Detection](https://github.com/jahn-chan/ML-Notebooks/blob/main/Pedestrain%20Detection.py)
* Prepared my own dataset for this task. 
* Utilized OpenCV and face_recognition library to detect and recognize faces in the picture.
* Applied Histogram of Oriented Gradients to the images whose output is then fed into Support Vector Machine to detect pedestrains.
* The script contains the code to detect the pedestrians in both images and real-time video.
* [Link to the article]()

### [Human Face Recognition](https://github.com/jahn-chan/ML-Notebooks/blob/main/Human%20Face%20Recognition.py)
* Prepared my own dataset for this task. 
* Utilized OpenCV and face_recognition library to detect and recognize faces in the picture.
* [Link to the article](https://data-flair.training/blogs/python-face-recognition)

### [Pedestrian Detection](https://github.com/jahn-chan/ML-Notebooks/blob/main/Pedestrain%20Detection.py)
* Prepared my own dataset for this task. 
* Utilized OpenCV and face_recognition library to detect and recognize faces in the picture.
* Applied Histogram of Oriented Gradients to the images whose output is then fed into Support Vector Machine to detect pedestrains.
* The script contains the code to detect the pedestrians in both images and real-time video.
* [Link to the article]()

### [Handwritten Text Generation using GANs](https://github.com/jahn-chan/Handwritten-Characters-Generation-using-GANs)
* Head over to the repo.
* [Link to the article](https://docs.google.com/document/d/1y2fB_XKXmNJbc2aTOipTgmPPrkA2bcEsrq7_kzg0NfI/edit?usp=sharing)

### [Image Caotioning](https://github.com/jahn-chan/Image-Caption-Generator)
* Head over to the repo.
* [Link to the article](https://docs.google.com/document/d/1R6sD3xU9-g9DijnASsIOhd-XcYd-g7QbWDBicO9_qgQ/edit?usp=sharing)

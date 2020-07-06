# Automatic-Essay-Scoring (AES)
Automated Essay Scoring (AES) is a tool for evaluating and scoring of essays written in response to specific prompts. It can be defined as the process of scoring written essays using computer programs. The process of automating the assessment process could be useful for both educators and learners since it encourages the iterative improvements of students' writings. 

## Why AES?
Automated grading if proven effective will not only reduce the time for assessment but comparing it with human scores will also make the score realistic. The project aims to develop an automated essay assessment system by use of machine learning techniques and Neural networks by classifying a corpus of textual entities into a small number of discrete categories, corresponding to possible grades.

## Dataset

The dataset we are using is ‘The Hewlett Foundation: Automated Essay Scoring Dataset’ by ASAP. You can find in the below link or download from the Dataset folder. 
 
https://www.kaggle.com/c/asap-aes/data 

## Proposed Model
The model is divided into 4 modules as follows:

1. **Data Preprocessing**

We began by doing some standard preprocessing steps like filling in null values and selecting valid features from the entire dataset after a thorough study.Next we plotted a graph to get a measure of the skewness of our data  and applied normalisation techniques to reduce this skewness.The next step involved cleaning the essays to make our training process easier for getting a better accuracy.To achieve this we removed all the  unnecessary symbols ,stop words and punctuations from our essays. To increase our accuracy even more we even planned to add some extra features like the number of sentences , number of words,number of characters, average word length etc. Moreover , we even worked on techniques like getting the noun ,verb ,adjective and adverb counts using parts of speech tagging as well as getting the total misspellings in an essay by comparison with a corpus.We applied various machine learning algorithms on this data as explained in the next section.

2. **Machine Learning**

For making our data ready to apply algorithms,we require one more step.Machine learning algorithms can not be applied on sentences or words,they can only be used upon numeric data.Our 
dataset has a field which has essays that need to be converted into a numeric form first in order to train it.To do this we use something known as a CountVectorizer.Now the CountVectorizer works by tokenizing a collection of text documents and returning an encoded vector with a length of the entire vocabulary along with an integer count for the number of times each word appeared in the document.After this step our data is finally ready for predictive modelling. 
 
Initially we applied machine learning algorithms like linear regression, SVR and Random Forest on the dataset without addition of features that were mentioned in the preprocessing section before. Our results were not really satisfactory as our mean squared error was quite high for all the above algorithms.After this initial evaluation, we added the extra features,applied CountVectorizer again on this modified dataset and applied the same three algorithms.There was a great improvement in the performance of all three algorithms especially Random forest for which the mean squared error reduced drastically. 
 
 
 3. **Applying Neural Networks**
 
Preprocessing steps for neural networks are different from preprocessing steps for machine learning algorithms. Our training data is fed into the Embedding Layer which is Word2Vec. Word2Vec is a shallow, two-layer neural network which is trained to reconstruct linguistic contexts of words. It takes as its input a large corpus of words and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located in close proximity to one another in the space. Word2Vec is a particularly computationally-efficient predictive model for learning word embeddings from raw text. Features from Word2Vec are fed into LSTM.  LSTM can learn which data in a sequence is important to keep or throw away. This largely helps in calculating scores from essays. Finally the Dense layer with output 1 predicts the score of each essay. 
 
 4. **Creation of web App**
 
 After training our model, the next step was to make our project available to users.For this purpose we planned to develop a web application for our model.To accomplish this we used the Flask framework to deploy our model. Flask is a popular Python web framework, meaning it is a thirdparty Python library used for developing web applications.Using Flask we were able to make an API which receives essay details through GUI and computes the predicted score value based on our model.The results can be shown by making a POST request . It receives JSON inputs, uses the trained model to make a prediction and returns that prediction in JSON format which can be accessed through the API endpoint. 
 

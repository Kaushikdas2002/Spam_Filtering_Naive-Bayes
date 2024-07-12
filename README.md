# Spam_Filtering_Naive-Bayes

# Introduction
This project aims to classify email messages as either spam or ham (not spam) using the Naive Bayes algorithm. The Naive Bayes classifier is a popular machine learning algorithm for text classification tasks due to its simplicity and effectiveness. By leveraging this algorithm, we can automatically filter out spam messages, thereby improving email management and user experience.  

# Dataset
The dataset used for this project is a collection of email messages labeled as either spam or ham. The dataset is typically structured in a CSV file with columns for the label (spam or ham) and the text of the email. In this example, we use a dataset from a publicly available source.

# Processes Involved
The project involves several key processes:  
1. **Data Loading**: Loading the dataset containing labeled email messages.  
2. **Data Preprocessing**: Cleaning and preprocessing the text data to prepare it for feature extraction.  
3. **Feature Extraction**: Converting text data into numerical features using count vectorizer.  
4. **Model Training**: Training a Naive Bayes classifier on the preprocessed and vectorized data.  
5. **Model Evaluation**: Evaluating the trained model using metrics such as accuracy, precision, recall, and F1 score.  
6. **Testing**: Testing the model with example messages to verify its performance.

# Steps
1. **Import Libraries**: Import necessary libraries such as pandas, sklearn, naive-bayes and others required for data processing, feature extraction, and model training.  
2. **Load Dataset**: Load the dataset containing emails/messages labeled as spam or ham from a CSV file.  
3. **Explore Dataset**: Perform initial exploration of the dataset to understand its structure and content.  
4. **Preprocess Data**: Clean and preprocess the text data.    
5. **Split Dataset**: Split the dataset into training and test sets using train_test_split from sklearn.  
6. **Feature Extraction**: Convert the text data into numerical features using count vectorizer from sklearn.  
7. **Train Naive Bayes Classifier**: Train a Naive Bayes classifier (MultinomialNB) on the training data.  
8. **Make Predictions**: Use the trained Naive Bayes model to make predictions on the test data.  
9. **Evaluate Model**: Evaluate the performance of the model using metrics such as accuracy, precision, recall, and F1 score. These metrics provide insights into the effectiveness of the classifier.

# Results
Accuracy: 98%  
Precision: 98%  

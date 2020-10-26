# Sentiment Classification (TFIDF, Word2Vec, BERT)


### HOW TO RUN:
The easiest way to run the .ipynb files is to load the files and training/testing files on google colab. Google Colab has most of the libraries already installed. Some libraries that are missing are already commented in the notebooks. Please uncomment and run that cells to download libraries and pretrained models.

### Libraries:
* numpy
* pandas
* nltk
* sklearn
* spacy
* gensim
* tensorflow (version 2 for ktrain)
* ktrain

### MODELS:
1)  TFIDF + ML Classifiers: 
	Please run the TFIDF.ipynb file on Google Colab to avoid dependency issues.
	Upload the training and testing files on google colab.
	Download nltk stopwords by uncommenting the code in second cell.
	
	CODE_EXPLANATION:
	1) Start with loading the libraries.
	2) Download and import stopwords.
	3) Read the training and testing datasets. Ensure you give the correct path to the excel files. You can upload the excel files on colab and update the file location. 
	4) Preprocess the datasets -  This  step involves lower casing the senteces, removing punctuation and stopwords. It uses get_tokens function to get tokens for each sentence.
	5) TFIDF Vectorizer -  Calculate the vectors for each sentence and apply fit and transform to the training dataset.
	6) ML Models -  Performed Support Vector Classifier and Random Forest Classifier to classify the testing dataset.
	
2)	WORD2VEC + ML Classifier (Gensim):
	Please run the Word2vec_gensim.ipynb file on Google Colab to avoid dependency issues. 
	Upload the training and testing files on google colab.
	Run the first cell and download google's pretrained model and nltk stopwords
	
	CODE_EXPLANATION:
	1) Start with loading the libraries.
	2) Download and import stopwords and spacy pretrained model.
	3) Read the training and testing datasets. Ensure you give the correct path to the excel files. You can upload the excel files on colab and update the file location. 
	4) Preprocess the datasets -  This  step involves lower casing the senteces, removing punctuation and stopwords. It uses get_tokens function to get tokens for each sentence.
	5) Load the pretrained model and update the vocabulary with training data. Train on the training dataset.
	6) Calculate the average  vectors for each sentence.
	7) Use Random Forest Classifier to identify the sentiments.
	
	
3) WORD2VEC + ML Classifiers (Spacy):
	Please run the Word2vec.ipynb file on Google Colab to avoid dependency issues. 
	Upload the training and testing files on google colab.
	Download nltk stopwords and load spacy model (en_core_web_lg) by uncommenting the code in second cell.
	
	CODE_EXPLANATION:
	1) Start with loading the libraries.
	2) Download and import stopwords and spacy pretrained model.
	3) Read the training and testing datasets. Ensure you give the correct path to the excel files. You can upload the excel files on colab and update the file location. 
	4) Preprocess the datasets -  This  step involves lower casing the senteces, removing punctuation and stopwords. It uses get_tokens function to get tokens for each sentence.
	5) Vector Representation -  Calculate the vectors for each sentence. Spacy calculates the vector representation using dov.vector function.
	6) Reshape the vector representations for input to machine learning models 
	7) ML Models -  Performed Support Vector Classifier and Random Forest Classifier to classify the testing dataset
	
4) BERT Model:
	Please run the Proposed_BERT.ipynb file on Google Colab to avoid dependency issues. 
	Please enure that you are using GPU so that it can train faster. Change the runtime to GPU in colab.
	Upload the training and testing files on google colab.
	Uncomment the first cell to download ktrain and stopwords.
	
	CODE_EXPLANATION:
	1) Download and import stopwords and ktrain.
	2) Install libraries
	3) Read the training and testing datasets. Ensure you give the correct path to the excel files. You can upload the excel files on colab and update the file location. 
	4) Keep 10% of training data as validation data for tuning the BERT model
	5) ktrain uses the BERT model which will be downloaded and preprocessed based on it's inbuilt preprocessing techniques.
	6) Build the model and train the model on training data.
	7) Get Predictions for testing data and generate classification results.
	
## Report
To get a detailed information about model comparisions and performance, refer to Report.pdf

# LT2222 V21 Assignment 2

Assignment 2 from the course LT2222 in the University of Gothenburg's winter 2021 semester.

Your name: Sarab Youssef


## Part 1 (Preprocessing):
In the reprocessing, I have converted the text file into a big list used to create a dataframe. I n this list I have converted all words to a lowercase and and alo lemmatized them and removed the punctuation. Unfortunately punctuation was not removed properly by using string.punctuation so I had to find another way which in my case was to remove all the POSs which had anything but alphabets in it. I have also tried removing stop words but that in my opinion not always a good idea because it might affect the model negatively and might lead to a big loss of information and that will give you a bad model. I have also noticed when using the nltk.stem WordNetLemmatizer, that it fails to lemmatize the verbs as it does take the POS tag into account, but it doesn't magically determine it, so I had to use a different lemmatizer when th pos tag is belongs to the verb tags. at the end only the selected rows where added to the list and then the dataframe.

## Part 2 (NE Features Selection):
I have added 5 features before and 5 after the NE, with taking into account they should be in the same sentence or else they will be replaced bey a start/end token (<SN> as a start token or </SN> as an end token where N is a range starts with 1 to 5). Once we use a token there is no coming back and we have to fill the next empty features with the next token in the tokens list respectively. The features should also not be an NE and therefor they will be replaced by tokens as well or will be skipped if they are to be immediately after the processed NE to get the other features. I have extracted the post features and prefeatures one at a time and then combined them to the big list (features), then created it the instances.

## Part 3 (Creating Table and Reduction):
In part 3 I after creating the table by using the top_freq features as the names of the columns and then counting the occurrences of each feature in each instance, I have reduced the dimensionality to dims=300 to avoid overfitting and improve the model.



## Part 5 (Confusion Matrix):
The prediction of the training data was obviously much better than the prediction of the testing data, as the model has become familiar with the data of the training data and has seen it before. In general I could say the model was not good because most of the classes were mostly wrongly classified. We see a better behaviour when classifying geo and gpe, but not the other classes. I have also noticed that tim and nat were almost zeroes for some reason. I tried using higher dimensionality (dims=500) in reduction to see if the results would get better but unfortunately the results mostly were classified as tim, geo and gpe respectively and the rest were almost all zeroes, so I kept using the 300 dims.


def auto_response_classifier(df_unlabeled, df_labeled):
    
    import pandas as pd
    global df_final
    
    # Verifying label balance
    print('The balance of the datset for automated responses: ', df_labeled['auto_labels'].mean())
    
    # Create training and testing splits
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(df_labeled['content'], df_labeled['auto_labels'])
    
    # Create TF-IDF vectorizer for classification task
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vect = TfidfVectorizer(min_df = 3, ngram_range = (1,3)).fit(X_train)
    X_train_vectorized = vect.transform(X_train)

    # Create LR model: 1 to 3 ngrams with a minimum frequency of 3
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train) # Fitting the model with the Tf-idf training data

    predictions = model.predict(vect.transform(X_test)) # Creating predictions on the testing set
    
    # Measuring the accuracy of the test predictions
    print('AUC: ', roc_auc_score(y_test, predictions))
    
    # Creating predictions now for the unlabelled dataframe   
    df_unlabeled['auto_labels'] = model.predict(vect.transform(df_unlabeled['content']))
    
    df_final = pd.DataFrame(pd.concat([df_unlabeled, df_labeled], axis = 0))
    df_final = df_final.iloc[:, 3:]

    return df_final

def preprocess_text(df_final):
    
    ''' df_final is the name of the dataframe and col is the the column of the df_final (should look like df_final['col'])
    
    This function does the following: 
    1.) Fills NA rows with the word empty in the content colulmn.
    2a.) Counts the number of words that are common in auto-reply response emails
    2.) uses nltk tokenizer to tokenize the words
    3.) Removes stop words from the new tokenized words
    4.) Uses the WordNet Lemmatizer to lemmatize the tokens
    5.) Removes punctuation from the tokens
    6.) lowercases all of the words
    7.) Returns a column in the df_final with 1-6 completed.
    '''
    # Merging in the legislator party Id's
    # Merging in the legislator Id's
    import pandas as pd
    leg = pd.read_csv('https://raw.githubusercontent.com/mawalz05/legislator_email/main/legislator_list.csv')
    df_final = pd.merge(df_final, leg, on = ['legislator_id'], how = 'left')
    
    # Start by importing stopwords and tokenize from nltk
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    # If the reply email contains the original content of the email, then remove the original content
    tmp = df_final['content'].str.split(r'[\<\[].+@gmail.com') # Creating a temporary holder for string split
    df_final['content2'] = tmp.apply(lambda x: x[0]) # Applying the string split to each row and only keeping the first split (x: x[0])

    # Determining if the Assistant is responding to an email or not (needs to be done prior to lowercasing)
    r1 = r'Assistant|Aide'
    r2 = r'(with|to)\s(the)?\s?(Senator|Congressman|Congresswoman|Representative|Sen|Rep)\.?'
    comb = '(%s|%s)' %(r1, r2)
    df_final['assistant'] = df_final['content2'].str.contains(comb, case = True, regex = True).astype(int)
    
    # lowercase all words
    df_final['content2'] = df_final['content2'].apply(str.lower)
    
    # Use the nltk tokenizer to tokenize the words and create a new column
    df_final['tokens'] = df_final['content2'].apply(word_tokenize)

    # Removing the stop words from the tokens
    df_final['tokens'] = df_final['tokens'].apply(lambda x: [word for word in x if word not in stopwords.words('english')])
    
    # Stemming the tokens for word count length
    p_stemmer = nltk.stem.PorterStemmer()
    df_final['stem_tokens'] = df_final['tokens'].apply(lambda x: [p_stemmer.stem(word) for word in x])
    
    # Removing long words and short words
    df_final['stem_tokens'] = df_final['stem_tokens'].apply(lambda x: [word for word in x if len(word) < 20 and len(word) >= 3])
    
    for i in range(len(df_final)):
        if df_final.loc[i]['party'] == ' D' or df_final.loc[i]['party'] == 'D':
            df_final.loc[[i],['party']] = 'D'
        if df_final.loc[i]['party'] == ' R' or df_final.loc[i]['party'] == 'R':
            df_final.loc[[i],['party']] = 'R'
        if df_final.loc[i]['party'] != 'D' and df_final.loc[i]['party'] != 'R':
            df_final.loc[[i], ['party']] = 'Third-Party'
    
    for i in range(len(df_final)):
        if df_final.loc[i]['message_type'] == 'null_right':
            df_final.loc[[i],['topic']] = 'General'
            df_final.loc[[i],['slant']] = 'Left-Wing'
            df_final.loc[[i],['author']] = 'Other'
        if df_final.loc[i]['message_type'] == 'null_left':
            df_final.loc[[i],['topic']] = 'General'
            df_final.loc[[i], ['slant']] = 'Right-Wing'
            df_final.loc[[i],['author']] = 'Other'
        if df_final.loc[i]['message_type'] == 'null':
            df_final.loc[[i],['topic']] = 'General'
            df_final.loc[[i], ['slant']] = 'Non-Partisan'
            df_final.loc[[i],['author']] = 'Other'            

    return df_final

def pre_analysis(df_final):
    
    import pandas as pd
    # Determining if the email has a personalized salutation
    names = r'^(\\ufeff)*[A-Za-z]*[\.]?\s?[A-za-z]*[\.]?\s?(margaret|thomas|patricia|brown|elizabeth|smith|ashley|jones|jessica|johnson|stephanie|davis|barbara|williams|nicole|wilson|crystal|lee|heather|taylor|lauren|lewis|amber|jackson|megan|harris|nancy|anderson|rachel|thompson|kimberly|robinson|christina|clark|mary|miller|karen|moore|brittany|walker|rebecca|hall|laura|allen|danielle|young|emily|king|samantha|wright|angela|hill)+[\,\!\:\-]+'
    df_final['pers_salutation'] = df_final['content2'].str.contains(names, case = False, regex = True).astype(int)
    
    # Finding the total token count for processed and stemmed tokens
    df_final['stem_length'] = df_final['stem_tokens'].apply(len)
    
    # Determining whether the email requests a response from the constituent
    r1 = r'(send|prove|provide)+\s(me|us)*\s*(with)?\s?(your|an)\s(permanent|phone|home|address|cell|mailing|telephone|residential|voting)+'
    r2 = r'are\syou\sa\s(resident|registered|voter)+'
    r3 = r'where\s.*(do)?\s?you\slive'
    r4 = r'(reply|respond)\s(back)?\s?with\syour\s(permanent|phone|home|address|cell|mailing|telephone|residential|voting)+'
    r5 = r'(reply|respond)\sto\sthis\s(message|email|e-mail)\swith\syour\s(permanent|phone|home|address|cell|mailing|telephone|residential|voting)+'
    r6 = r'what(\'s)?\s(is)?\s?your\s(residential|home)?\s?address'
    r7 = r'(reply|respond)\s(back)?\s?with\sthat\sinformation'
    r8 = r'ask\s(you)?\s?for\syour\s(permanent|phone|home|address|cell|mailing|telephone|residential|voting)'
    r9 = r'are\syou\sa\sresident'
    r10 = r'can\syou\s(please)?\s?(share|verify)\syour\s(home)?\s?(permanent|phone|home|address|cell|mailing|telephone|residential|voting)+'
    r11 = r'you\s(live|reside)\sin(.)*\s?(county|district)*'
    r12 = r'(advise|let)\s(me|us)?\s?(know|of|with)\syour\s(full|complete)?\s?(permanent|phone|home|address|cell|mailing|telephone|residential|voting)+'
    r13 = r'(town|city)\s(you)?\s?(are)?\s?(from)?\s?in\s(my|the|our)\s(district)'
    
    full_regex = '(%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s)' %(r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13)

    df_final['response_checks'] = df_final['content'].str.contains(full_regex, case = False, regex = True).astype(int)
          
    return df_final

def analysis(df_final):
    
    import pandas as pd
    # Filtering for automated responses
    df = df_final[df_final['auto_labels'] == 0]
    
    # Counting how many total responses there are by topic, slant, and author
    topic = df.groupby('topic')['topic'].count()
    slant = df.groupby('slant')['slant'].count()
    author = df.groupby('author')['author'].count()
    
    # Calculating how many responses had personal salutations by topic, slant, and author
    pers_sal_topic = df.groupby('topic')['pers_salutation'].sum()
    pers_sal_slant = df.groupby('slant')['pers_salutation'].sum()
    pers_sal_author = df.groupby('author')['pers_salutation'].sum()

    # Calculating how many responses had response checks by topic, slant, and author
    response_topic = df.groupby('topic')['response_checks'].sum()
    response_slant = df.groupby('slant')['response_checks'].sum()
    response_author = df.groupby('author')['response_checks'].sum()

    # Calculating how many assistants responded to the emails by topic, slant, and author
    assistant_topic = df.groupby('topic')['assistant'].sum()
    assistant_slant = df.groupby('slant')['assistant'].sum()
    assistant_author = df.groupby('author')['assistant'].sum()

    # Calculating the mean and sd length of responses by topic, slant, and author
    stem_topic = df.groupby('topic')['stem_length'].mean()
    stem_slant = df.groupby('slant')['stem_length'].mean()
    stem_author = df.groupby('author')['stem_length'].mean()

    stem_topic_sd = df.groupby('topic')['stem_length'].std()
    stem_slant_sd = df.groupby('slant')['stem_length'].std()
    stem_author_sd = df.groupby('author')['stem_length'].std()
    

    # Creating individual data frames where these counts, sums, and means are combined by stacking
    topics = pd.DataFrame(pd.concat([topic, pers_sal_topic, response_topic, assistant_topic, stem_topic, stem_topic_sd], axis = 1))
    slants = pd.DataFrame(pd.concat([slant, pers_sal_slant, response_slant, assistant_slant, stem_slant, stem_slant_sd], axis = 1))
    authors = pd.DataFrame(pd.concat([author, pers_sal_author, response_author, assistant_author, stem_author, stem_author_sd], axis = 1))
    
    topics.columns = ['count', 'pers_salutation', 'response_checks', 'assistant','stem_length_mean','stem_length_sd']
    slants.columns = ['count', 'pers_salutation', 'response_checks', 'assistant','stem_length_mean','stem_length_sd']
    authors.columns = ['count', 'pers_salutation', 'response_checks', 'assistant','stem_length_mean','stem_length_sd']
    
#    #Making sure the names do not align for the stem_length means and standard deviations
#     cols = []
#     count = 1
#     for column in authors.columns:
#         if column == 'stem_length':
#             cols.append(f'stem_length_{count}') # stem_length_1 is the mean, stem_length_2 is the stdev.
#             count+=1
#             continue
#         cols.append(column)
#     authors.columns = cols

    # Creating proportions of each category by dividing the sums of variables by the total count of each topic, slant, author
    topics['pers_sal_prop'] = topics['pers_salutation']/topics['count']
    topics['response_prop'] = topics['response_checks']/topics['count']
    topics['assistant_prop'] = topics['assistant']/topics['count']

    slants['pers_sal_prop'] = slants['pers_salutation']/slants['count']
    slants['response_prop'] = slants['response_checks']/slants['count']
    slants['assistant_prop'] = slants['assistant']/slants['count']

    authors['pers_sal_prop'] = authors['pers_salutation']/authors['count']
    authors['response_prop'] = authors['response_checks']/authors['count']
    authors['assistant_prop'] = authors['assistant']/authors['count']

    ##################################################################################################
    # Repeating the steps above but this time grouping by both topics and slants
    topic_slants = df.groupby(['topic', 'slant'])['topic'].count()
    pers_sal_top_slant = df.groupby(['topic', 'slant'])['pers_salutation'].sum()
    response_top_slant = df.groupby(['topic', 'slant'])['response_checks'].sum()
    assistant_top_slant = df.groupby(['topic', 'slant'])['assistant'].sum()
    stem_top_slant = df.groupby(['topic', 'slant'])['stem_length'].mean()
    stem_top_slant_sd = df.groupby(['topic', 'slant'])['stem_length'].std()

    topics_slants = pd.DataFrame(pd.concat([topic_slants, pers_sal_top_slant, response_top_slant,  
                                            assistant_top_slant, stem_top_slant, stem_top_slant_sd], axis = 1))
    
    topics_slants.columns = ['count', 'pers_salutation', 'response_checks', 'assistant','stem_length_mean','stem_length_sd']

    topics_slants['pers_sal_prop'] = topics_slants['pers_salutation']/topics_slants['count']
    topics_slants['response_prop'] = topics_slants['response_checks']/topics_slants['count']
    topics_slants['assistant_prop'] = topics_slants['assistant']/topics_slants['count']

    ############################################################################################
    # Repeating the steps above but this time grouping by both topics and authors
    topic_auth = df.groupby(['topic', 'author'])['topic'].count()
    pers_sal_top_auth = df.groupby(['topic', 'author'])['pers_salutation'].sum()
    response_top_auth = df.groupby(['topic', 'author'])['response_checks'].sum()
    assistant_top_auth = df.groupby(['topic', 'author'])['assistant'].sum()
    stem_top_auth = df.groupby(['topic', 'author'])['stem_length'].mean()
    stem_top_auth_sd = df.groupby(['topic', 'author'])['stem_length'].std()

    topics_authors = pd.DataFrame(pd.concat([topic_auth, pers_sal_top_auth, response_top_auth, 
                                            assistant_top_auth, stem_top_auth, stem_top_auth_sd], axis = 1))
    
    topics_authors.columns = ['count', 'pers_salutation', 'response_checks', 'assistant','stem_length_mean','stem_length_sd']

    topics_authors['pers_sal_prop'] = topics_authors['pers_salutation']/topics_authors['count']
    topics_authors['response_prop'] = topics_authors['response_checks']/topics_authors['count']
    topics_authors['assistant_prop'] = topics_authors['assistant']/topics_authors['count']
    
    ############################################################################################
    # Repeating the steps above but this time grouping by both slants and authors
    slant_auth = df.groupby(['slant', 'author'])['topic'].count()
    pers_sal_slant_auth = df.groupby(['slant', 'author'])['pers_salutation'].sum()
    response_slant_auth = df.groupby(['slant', 'author'])['response_checks'].sum()
    assistant_slant_auth = df.groupby(['slant', 'author'])['assistant'].sum()
    stem_slant_auth = df.groupby(['slant', 'author'])['stem_length'].mean()
    stem_slant_auth_sd = df.groupby(['slant', 'author'])['stem_length'].std()

    slants_authors = pd.DataFrame(pd.concat([slant_auth, pers_sal_slant_auth, response_slant_auth, 
                                            assistant_slant_auth, stem_slant_auth, stem_slant_auth_sd], axis = 1))
    
    slants_authors.columns = ['count', 'pers_salutation', 'response_checks', 'assistant','stem_length_mean','stem_length_sd']

    slants_authors['pers_sal_prop'] = slants_authors['pers_salutation']/slants_authors['count']
    slants_authors['response_prop'] = slants_authors['response_checks']/slants_authors['count']
    slants_authors['assistant_prop'] = slants_authors['assistant']/slants_authors['count']
    
    ############################################################################################
    # Repeating the steps above but this time grouping by both slants and party
    slant_party = df.groupby(['slant', 'party'])['topic'].count()
    pers_sal_slant_party = df.groupby(['slant', 'party'])['pers_salutation'].sum()
    response_slant_party = df.groupby(['slant', 'party'])['response_checks'].sum()
    assistant_slant_party = df.groupby(['slant', 'party'])['assistant'].sum()
    stem_slant_party = df.groupby(['slant', 'party'])['stem_length'].mean()
    stem_slant_party_sd = df.groupby(['slant', 'party'])['stem_length'].std()

    slants_party = pd.DataFrame(pd.concat([slant_party, pers_sal_slant_party, response_slant_party, 
                                            assistant_slant_party, stem_slant_party, stem_slant_party_sd], axis = 1))
    
    slants_party.columns = ['count', 'pers_salutation', 'response_checks', 'assistant','stem_length_mean','stem_length_sd']

    slants_party['pers_sal_prop'] = slants_party['pers_salutation']/slants_party['count']
    slants_party['response_prop'] = slants_party['response_checks']/slants_party['count']
    slants_party['assistant_prop'] = slants_party['assistant']/slants_party['count']

    ############################################################################################
    # Repeating the steps above but this time grouping by topics, slants and authors
    topic_slants_auth = df.groupby(['topic', 'slant', 'author'])['topic'].count()
    pers_sal_top_slant_auth = df.groupby(['topic', 'slant', 'author'])['pers_salutation'].sum()
    response_top_slant_auth = df.groupby(['topic', 'slant', 'author'])['response_checks'].sum()
    assistant_top_slant_auth = df.groupby(['topic', 'slant', 'author'])['assistant'].sum()
    stem_top_slant_auth = df.groupby(['topic', 'slant', 'author'])['stem_length'].mean()
    stem_top_slant_auth_sd = df.groupby(['topic', 'slant', 'author'])['stem_length'].std()

    topics_slants_authors = pd.DataFrame(pd.concat([topic_slants_auth, pers_sal_top_slant_auth, 
                                                    response_top_slant_auth, assistant_top_slant_auth, stem_top_slant_auth,
                                                    stem_top_slant_auth_sd], axis = 1))   
    
    topics_slants_authors.columns = ['count', 'pers_salutation', 'response_checks', 'assistant','stem_length_mean','stem_length_sd']
    
    topics_slants_authors['pers_sal_prop'] = topics_slants_authors['pers_salutation']/topics_slants_authors['count']
    topics_slants_authors['response_prop'] = topics_slants_authors['response_checks']/topics_slants_authors['count']
    topics_slants_authors['assistant_prop'] = topics_slants_authors['assistant']/topics_slants_authors['count']
    
    ############################################################################################
    # Repeating the steps above but this time grouping by topics, slants and party
    topic_slants_party = df.groupby(['topic', 'slant', 'party'])['topic'].count()
    pers_sal_top_slant_party = df.groupby(['topic', 'slant', 'party'])['pers_salutation'].sum()
    response_top_slant_party = df.groupby(['topic', 'slant', 'party'])['response_checks'].sum()
    assistant_top_slant_party = df.groupby(['topic', 'slant', 'party'])['assistant'].sum()
    stem_top_slant_party = df.groupby(['topic', 'slant', 'party'])['stem_length'].mean()
    stem_top_slant_party_sd = df.groupby(['topic', 'slant', 'party'])['stem_length'].std()

    topics_slants_party = pd.DataFrame(pd.concat([topic_slants_party, pers_sal_top_slant_party, 
                                                    response_top_slant_party, assistant_top_slant_party, stem_top_slant_party,
                                                    stem_top_slant_party_sd], axis = 1))   
    
    topics_slants_party.columns = ['count', 'pers_salutation', 'response_checks', 'assistant','stem_length_mean','stem_length_sd']
    
    topics_slants_party['pers_sal_prop'] = topics_slants_party['pers_salutation']/topics_slants_party['count']
    topics_slants_party['response_prop'] = topics_slants_party['response_checks']/topics_slants_party['count']
    topics_slants_party['assistant_prop'] = topics_slants_party['assistant']/topics_slants_party['count']

    ############################################################################################
    # Repeating the steps above but this time grouping by authors, slants and party
    auth_slants_party = df.groupby(['author', 'slant', 'party'])['topic'].count()
    pers_sal_auth_slant_party = df.groupby(['author', 'slant', 'party'])['pers_salutation'].sum()
    response_auth_slant_party = df.groupby(['author', 'slant', 'party'])['response_checks'].sum()
    assistant_auth_slant_party = df.groupby(['author', 'slant', 'party'])['assistant'].sum()
    stem_auth_slant_party = df.groupby(['author', 'slant', 'party'])['stem_length'].mean()
    stem_auth_slant_party_sd = df.groupby(['author', 'slant', 'party'])['stem_length'].std()

    auth_slants_party = pd.DataFrame(pd.concat([auth_slants_party, pers_sal_auth_slant_party, 
                                                    response_auth_slant_party, assistant_auth_slant_party, stem_auth_slant_party,
                                                    stem_auth_slant_party_sd], axis = 1))   
    
    auth_slants_party.columns = ['count', 'pers_salutation', 'response_checks', 'assistant','stem_length_mean','stem_length_sd']
    
    auth_slants_party['pers_sal_prop'] = auth_slants_party['pers_salutation']/auth_slants_party['count']
    auth_slants_party['response_prop'] = auth_slants_party['response_checks']/auth_slants_party['count']
    auth_slants_party['assistant_prop'] = auth_slants_party['assistant']/auth_slants_party['count']

    ############################################################################################
    # Repeating the steps above but this time grouping by topics, slants, authors and party
    topic_slants_auth_party = df.groupby(['topic', 'slant', 'author','party'])['topic'].count()
    pers_sal_top_slant_auth_party = df.groupby(['topic', 'slant', 'author', 'party'])['pers_salutation'].sum()
    response_top_slant_auth_party = df.groupby(['topic', 'slant', 'author', 'party'])['response_checks'].sum()
    assistant_top_slant_auth_party = df.groupby(['topic', 'slant', 'author', 'party'])['assistant'].sum()
    stem_top_slant_auth_party = df.groupby(['topic', 'slant', 'author', 'party'])['stem_length'].mean()
    stem_top_slant_auth_party_sd = df.groupby(['topic', 'slant', 'author', 'party'])['stem_length'].std()

    topics_slants_auth_party = pd.DataFrame(pd.concat([topic_slants_auth_party, pers_sal_top_slant_auth_party, 
                                                    response_top_slant_auth_party, assistant_top_slant_auth_party, stem_top_slant_auth_party,
                                                    stem_top_slant_auth_party_sd], axis = 1))   
    
    topics_slants_auth_party.columns = ['count', 'pers_salutation', 'response_checks', 'assistant','stem_length_mean','stem_length_sd']
    
    topics_slants_auth_party['pers_sal_prop'] = topics_slants_auth_party['pers_salutation']/topics_slants_auth_party['count']
    topics_slants_auth_party['response_prop'] = topics_slants_auth_party['response_checks']/topics_slants_auth_party['count']
    topics_slants_auth_party['assistant_prop'] = topics_slants_auth_party['assistant']/topics_slants_auth_party['count']

    ##############################################################################################  
    from datetime import datetime
    import os
    cwd = os.getcwd()
    
    # Obtain timestamp in a readable format
    to_csv_timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(cwd + '\\' + to_csv_timestamp + "_analysis_response.xlsx", engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    topics.to_excel(writer, sheet_name='topics')
    slants.to_excel(writer, sheet_name='slants')
    authors.to_excel(writer, sheet_name='authors')

    topics_slants.to_excel(writer, sheet_name='topics_slants')
    topics_authors.to_excel(writer, sheet_name='topics_authors')
    slants_authors.to_excel(writer, sheet_name='slants_authors')
    
    slants_party.to_excel(writer, sheet_name = 'slants_party')
    auth_slants_party.to_excel(writer, sheet_name='auth_slants_party')
    topics_slants_party.to_excel(writer, sheet_name='topics_slants_party')

    topics_slants_authors.to_excel(writer, sheet_name='topics_slants_authors')
    topics_slants_auth_party.to_excel(writer, sheet_name = 'topics_slants_auth_party')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    return authors

def comp_tests(df, group1, group2):
    import pandas as pd
    import numpy as np
        
    phat = [(df['pers_salutation'][group1]+ df['pers_salutation'][group2])/(df['count'][group1] + df['count'][group2]),
           (df['response_checks'][group1]+ df['response_checks'][group2])/(df['count'][group1] + df['count'][group2]),
           (df['assistant'][group1]+ df['assistant'][group2])/(df['count'][group1] + df['count'][group2]),
           df['stem_length_mean'][group1] - df['stem_length_mean'][group2]]

    se = [np.sqrt((df['pers_salutation'][group1]+ df['pers_salutation'][group2])/(df['count'][group1] + df['count'][group2])*
                            (1 - (df['pers_salutation'][group1]+ df['pers_salutation'][group2])/(df['count'][group1] + df['count'][group2]))*
                            (1/df['pers_salutation'][group1] + 1/df['pers_salutation'][group2])),
         np.sqrt((df['response_checks'][group1]+ df['response_checks'][group2])/(df['count'][group1] + df['count'][group2])*
                            (1 - (df['response_checks'][group1]+ df['response_checks'][group2])/(df['count'][group1] + df['count'][group2]))*
                            (1/df['response_checks'][group1] + 1/df['response_checks'][group2])),
          np.sqrt((df['assistant'][group1]+ df['assistant'][group2])/(df['count'][group1] + df['count'][group2])*
                            (1 - (df['assistant'][group1]+ df['assistant'][group2])/(df['count'][group1] + df['count'][group2]))*
                            (1/df['assistant'][group1] + 1/df['assistant'][group2])),
         np.sqrt(df['stem_length_sd'][group1]**2/df['count'][group1] + df['stem_length_sd'][group2]**2/df['count'][group2])]

    z = [(df['pers_sal_prop'][group1] - df['pers_sal_prop'][group2])/se[0],
        (df['response_prop'][group1] - df['response_prop'][group2])/se[1],
        (df['assistant_prop'][group1] - df['assistant_prop'][group2])/se[2],
        phat[3]/se[3]]

    upper = [(df['pers_sal_prop'][group1] - df['pers_sal_prop'][group2]) + 1.96*se[0],
            (df['response_prop'][group1] - df['response_prop'][group2]) + 1.96*se[1],
            (df['assistant_prop'][group1] - df['assistant_prop'][group2]) + 1.96*se[2],
            phat[3] + 1.96*se[3]]

    lower = [(df['pers_sal_prop'][group1] - df['pers_sal_prop'][group2]) - 1.96*se[0],
            (df['response_prop'][group1] - df['response_prop'][group2]) - 1.96*se[1],
            (df['assistant_prop'][group1] - df['assistant_prop'][group2]) - 1.96*se[2],
            phat[3] - 1.96*se[3]]

    stats = pd.DataFrame(np.vstack((phat, se, z, upper, lower)), 
                                 columns = ['salutation', 'response_check', 'assistant', 'stem_length']).set_index(pd.Index(['phat', 'se',
                                                                                                             'z', 'upper',
                                                                                                             'lower']))
    print(stats)
    
    from datetime import datetime
    import os
    cwd = os.getcwd()
    
    # Obtain timestamp in a readable format
    to_csv_timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(cwd + '\\' + to_csv_timestamp + "_ttest_response.xlsx", engine='xlsxwriter')

    stats.to_excel(writer, sheet_name='ttest')
    writer.save()

def clean_analysis(df):
    for i in range(len(df)):
        if df.loc[i]['message_type'] == 'null_right':
            df.loc[[i],['topic']] = 'General'
            df.loc[[i],['slant']] = 'Right-Wing'
            df.loc[[i],['author']] = 'Other'
        if df.loc[i]['message_type'] == 'null_left':
            df.loc[[i],['topic']] = 'General'
            df.loc[[i], ['slant']] = 'Left-Wing'
            df.loc[[i],['author']] = 'Other'
        if df.loc[i]['message_type'] == 'null':
            df.loc[[i],['topic']] = 'General'
            df.loc[[i], ['slant']] = 'Non-Partisan'
            df.loc[[i],['author']] = 'Other' 
    
    for i in range(len(df)):
        if df.loc[i]['party'] == ' D' or df.loc[i]['party'] == 'D':
            df.loc[[i],['party']] = 'D'
        if df.loc[i]['party'] == ' R' or df.loc[i]['party'] == 'R':
            df.loc[[i],['party']] = 'R'
        if df.loc[i]['party'] != 'D' and df.loc[i]['party'] != 'R':
            df.loc[[i], ['party']] = 'Third-Party'
            
    return df


def merged_analysis(df):
    
    import pandas as pd
    import numpy as np
    # Filtering for automated responses
    df = df[df['auto_labels'] == 0]
    
    # Counting how many total responses there are by topic, slant, and author
    topic = df.groupby('topic')['topic'].count()
    slant = df.groupby('slant')['slant'].count()
    author = df.groupby('author')['author'].count()
    
    response_topic = df.groupby('topic')['response'].sum()
    response_slant = df.groupby('slant')['response'].sum()
    response_author = df.groupby('author')['response'].sum()
    
    topics = pd.DataFrame(pd.concat([topic, response_topic], axis = 1))
    slants = pd.DataFrame(pd.concat([slant, response_slant], axis = 1))
    authors = pd.DataFrame(pd.concat([author, response_author], axis = 1))
    
    topics.columns = ['count', 'response']
    slants.columns = ['count', 'response']
    authors.columns = ['count', 'response']
    
    topics['response_prop'] = topics['response']/topics['count']
    slants['response_prop'] = slants['response']/slants['count']
    authors['response_prop'] = authors['response']/authors['count']
 

    topic_slants = df.groupby(['topic', 'slant'])['topic'].count()
    response_topic_slants = df.groupby(['topic','slant'])['response'].sum()
    
    topics_slants = pd.DataFrame(pd.concat([topic_slants, response_topic_slants], axis = 1))
    
    topics_slants.columns = ['count', 'response']
    
    topics_slants['response_prop'] = topics_slants['response']/topics_slants['count']
    

    topic_authors = df.groupby(['topic', 'author'])['topic'].count()
    response_topic_authors = df.groupby(['topic','author'])['response'].sum()
    
    topics_authors = pd.DataFrame(pd.concat([topic_authors, response_topic_authors], axis = 1))
    
    topics_authors.columns = ['count', 'response']
    
    topics_authors['response_prop'] = topics_authors['response']/topics_authors['count']
    
    
    slant_authors = df.groupby(['slant', 'author'])['topic'].count()
    response_slant_authors = df.groupby(['slant','author'])['response'].sum()
    
    slants_authors = pd.DataFrame(pd.concat([slant_authors, response_slant_authors], axis = 1))
    
    slants_authors.columns = ['count', 'response']
    
    slants_authors['response_prop'] = slants_authors['response']/slants_authors['count']
    
    
    slant_party = df.groupby(['slant', 'party'])['topic'].count()
    response_slant_party = df.groupby(['slant','party'])['response'].sum()
    
    slants_party = pd.DataFrame(pd.concat([slant_party, response_slant_party], axis = 1))
    
    slants_party.columns = ['count', 'response']
    
    slants_party['response_prop'] = slants_party['response']/slants_party['count']
    
    
    topic_slants_authors = df.groupby(['topic', 'slant', 'author'])['topic'].count()
    response_topic_slants_authors = df.groupby(['topic','slant', 'author'])['response'].sum()
    
    topics_slants_authors = pd.DataFrame(pd.concat([topic_slants_authors, response_topic_slants_authors], axis = 1))
    
    topics_slants_authors.columns = ['count', 'response']
    
    topics_slants_authors['response_prop'] = topics_slants_authors['response']/topics_slants_authors['count']

    
    topic_slants_party = df.groupby(['topic', 'slant', 'party'])['topic'].count()
    response_topic_slants_party = df.groupby(['topic','slant', 'party'])['response'].sum()
    
    topics_slants_party = pd.DataFrame(pd.concat([topic_slants_party, response_topic_slants_party], axis = 1))
    
    topics_slants_party.columns = ['count', 'response']
    
    topics_slants_party['response_prop'] = topics_slants_party['response']/topics_slants_party['count']

    
    authors_slants_party = df.groupby(['author', 'slant', 'party'])['topic'].count()
    response_authors_slants_party = df.groupby(['author','slant', 'party'])['response'].sum()
    
    authors_slants_party = pd.DataFrame(pd.concat([authors_slants_party, response_authors_slants_party], axis = 1))
    
    authors_slants_party.columns = ['count', 'response']
    
    authors_slants_party['response_prop'] = authors_slants_party['response']/authors_slants_party['count']
 

    topic_authors_slants_party = df.groupby(['topic','author', 'slant', 'party'])['topic'].count()
    response_topic_authors_slants_party = df.groupby(['topic','author','slant', 'party'])['response'].sum()
    
    topic_authors_slants_party = pd.DataFrame(pd.concat([topic_authors_slants_party, response_topic_authors_slants_party], axis = 1))
    
    topic_authors_slants_party.columns = ['count', 'response']
    
    topic_authors_slants_party['response_prop'] = topic_authors_slants_party['response']/topic_authors_slants_party['count']
    
    import os
    cwd = os.getcwd()    
    from datetime import datetime
    
    # Obtain timestamp in a readable format
    to_csv_timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(cwd + '\\' + to_csv_timestamp + "_analysis_all.xlsx", engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    topics.to_excel(writer, sheet_name='topics')
    slants.to_excel(writer, sheet_name='slants')
    authors.to_excel(writer, sheet_name='authors')

    topics_slants.to_excel(writer, sheet_name='topics_slants')
    topics_authors.to_excel(writer, sheet_name='topics_authors')
    slants_authors.to_excel(writer, sheet_name='slants_authors')
    slants_party.to_excel(writer, sheet_name = 'slants_party')
    
    topics_slants_party.to_excel(writer, sheet_name = 'topic_slants_party')
    authors_slants_party.to_excel(writer, sheet_name = 'authors_slants_party')
    topics_slants_authors.to_excel(writer, sheet_name='topics_slants_authors')
    
    topic_authors_slants_party.to_excel(writer, sheet_name = 'topic_authors_slants_party')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    return authors 


def comp_tests_merge(df, group1, group2):

    import pandas as pd
    import numpy as np
        
    phat = (df['response'][group1]+ df['response'][group2])/(df['count'][group1] + df['count'][group2])

    se = np.sqrt((df['response'][group1]+ df['response'][group2])/(df['count'][group1] + df['count'][group2])*
                            (1 - (df['response'][group1]+ df['response'][group2])/(df['count'][group1] + df['count'][group2]))*
                            (1/df['response'][group1] + 1/df['response'][group2]))

    z = (df['response_prop'][group1] - df['response_prop'][group2])/se

    upper = (df['response_prop'][group1] - df['response_prop'][group2]) + 1.96*se

    lower = (df['response_prop'][group1] - df['response_prop'][group2]) - 1.96*se
         
    print('phat: ', phat)
    print('se: ', se)
    print('z: ', z)
    print('upper: ', upper)
    print('lower: ', lower)
    
    s_list = [[phat, se, z, upper, lower]]
    
    stats = pd.DataFrame(s_list, columns = ['phat', 'se', 'z', 'upper','lower'])
 
    import os
    cwd = os.getcwd()    
    from datetime import datetime
    
    # Obtain timestamp in a readable format
    to_csv_timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(cwd + '\\' + to_csv_timestamp + "_ttest_all.xlsx", engine='xlsxwriter')

    stats.to_excel(writer, sheet_name='ttest')
    writer.save()    
    
def logit_output(df):  
    import numpy as np
    import statsmodels.api as sm
    import pandas as pd
    from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
    
    # Filtering out for automated responses
    df = df[df['auto_labels'] == 0]

    # One-hot encoding
    author = df['author']
    author = author.to_numpy(dtype = str)
    one_hot = LabelBinarizer()
    author = one_hot.fit_transform(author)
    author_names = one_hot.classes_
    author = pd.DataFrame(author)
    author.columns = author_names

    # One-hot encoding slants
    slant = df['slant']
    slant = slant.to_numpy(dtype = str)
    slant = one_hot.fit_transform(slant)
    slant_names = one_hot.classes_
    slant = pd.DataFrame(slant)
    slant.columns = slant_names

    # One-hot encoding topics
    topic = df['topic']
    topic = topic.to_numpy(dtype = str)
    topic = one_hot.fit_transform(topic)
    topic_names = one_hot.classes_
    topic = pd.DataFrame(topic)
    topic.columns = topic_names

    # One-hot encoding parties
    party = df['party']
    party = party.to_numpy(dtype = str)
    party = one_hot.fit_transform(party)
    party_names = one_hot.classes_
    party = pd.DataFrame(party)
    party.columns = party_names
    
    squire = df['squire_score2015']
    squire = np.nan_to_num(squire) # This fixes the error with Nan or infinity values
    squire = pd.DataFrame(squire)
    
    # Creating a new dataframe with the dummy vectors
    X = pd.concat([author, slant, topic, party, squire], axis = 1)
    # Creating a vector for the dv
    y = list(df['response'])

    # Dropping one category from each column for comparison
    X = X.drop(['Human','General','D', 'Left-Wing'], axis = 1)

    # Interacting Rightwing and Republican
    X['Right_Wing * R'] = X['Right-Wing']*X['R']

    # Running Logistic Regression
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression().fit(X,y)

    logit_model = sm.Logit(y, X)
    result = logit_model.fit()
    print(result.summary())
    
    import os
    cwd = os.getcwd()    
    from datetime import datetime
    
    # Obtain timestamp in a readable format
    to_text_timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')    

    with open(cwd + '\\' + to_text_timestamp + '_logit_regression.txt', 'w' , encoding='utf-8') as text_file:
        text_file.write(result.summary().as_text())
        text_file.close()

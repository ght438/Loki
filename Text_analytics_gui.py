# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:14:10 2015

@author: GHT438
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:03:31 2015

@author: ght438
"""
#import textblob
import textblob as tb
import re
import pandas as pd
from nltk import tokenize
from nltk import FreqDist
import numpy as np
from string import punctuation
import nltk
from nltk.corpus import stopwords
from pandas import ExcelWriter
import parse_and
#import sys
import operator

def model_parsing():
    model_df=pd.read_csv('Text_Model_twitter_v1.csv')
    model_df=model_df.fillna('mv-5')
    for index, row_k  in enumerate(model_df['Keywords']):
        row_k=parse_and.parse(str(row_k))
        model_df.Keywords[index] = row_k
    for index, row_k  in enumerate(model_df['Keywords_and_1']):
        row_k=parse_and.parse(str(row_k))
        model_df.Keywords_and_1[index] = row_k
    for index, row_k  in enumerate(model_df['Keywords_and_2']):
        row_k=parse_and.parse(str(row_k))
        model_df.Keywords_and_2[index] = row_k
    for index, row_e in enumerate(model_df['Exclusion']):
        row_e=parse_and.parse(str(row_e))
        model_df.Exclusion[index] = row_e
    for index, row_e in enumerate(model_df['Attribute_Or_keyword']):
        row_e=parse_and.parse(str(row_e))
        model_df.Attribute_Or_keyword[index] = row_e
    for index, row_e in enumerate(model_df['Attribute_And_keyword']):
        row_e=parse_and.parse(str(row_e))
        model_df.Attribute_And_keyword[index] = row_e
    model_df.to_csv('Model_modified_twitter_test.csv',index=False)
    
"""Function to Exclude a category based on pre-defined keywords criteria"""
def Exclusion(line,k,model_df):
    exc_flag=0
    Exclusion_df = model_df
    Exclusion_df.Exclusion=Exclusion_df.Exclusion.fillna(0)
    Exclusion_category=Exclusion_df['Category']  
    Exclusion_separated =Exclusion_df['Exclusion'].astype(str).str.split(',')
    Exclusion_dict = dict(zip(Exclusion_category, Exclusion_separated))  
    a=0    
    for eword_temp in Exclusion_dict[k]: 
        words_separated=eword_temp.split(' AND ')
        for eword in words_separated:
            eword=eword.lower()
            find=re.compile('\+')   
            replace='.+'  
            eword = find.sub(replace,eword)
            find=re.compile('\?')   
            replace='.?'
            eword = find.sub(replace,eword)
            find=re.compile('\"')   
            replace=''
            eword = find.sub(replace,eword)
            if  re.search("\\b%s\\b"%eword, line):
                a=a+1
        if a == len(words_separated):
            exc_flag=1
        a=0
    if exc_flag==1:
        return True
    else:
        return False

def Attribute_check(uid,line,sentiment_score,k,model_df,attribute_column,original_line):
    Attribute_df = model_df
    Attribute_category=Attribute_df['Category']  
    Attrbute_separated =Attribute_df[attribute_column].astype(str).str.split(',')
    Attribute_dict = dict(zip(Attribute_category, Attrbute_separated))
    temp_result_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
    data_df=pd.read_csv('twitter.csv')
    data_df=data_df.fillna('mv-a2')
    cat_asgign_ind=0
    a=0
    for values in Attribute_dict[k]:
        if cat_asgign_ind == 0 and values != 'mv-5':
            values_separated=values.split(' AND ')
            for value in values_separated:
                fields = re.split(r'(==|>|<|!=|>=|<=)\s*', value)
                try:     
                    attribute_name=fields[0].strip()
                    attribute_actual= data_df['%s'%attribute_name].loc[data_df['Unique_Id'] == uid].values
                except KeyError:
                    #sys.exit("Specified attribute, %s , from model does not exist in the datafile, Please check and rerun the classification"%attribute_name)
                    print "Specified attribute does not exist in the datafile, Please check and rerun the classification"                
                if attribute_actual[0] != 'mv-a2':
                    try:
                        ops = {'==' : operator.eq,'!=' : operator.ne,'<=' : operator.le,'>=' : operator.ge,'>'  : operator.gt,'<'  : operator.lt}
                        find=re.compile('\'')   
                        replace=''
                        check_value = find.sub(replace, fields[2])
                        try:
                            if ops[fields[1]](float(attribute_actual[0]),float(check_value)):
                                a=a+1
                        except:
                            if ops['=='](str(check_value), str(attribute_actual[0])):
                                a=a+1
                    except Exception as e:
                        print "Exceptip %s generated"%e                 
            if a == len(values_separated):
                temp_df = pd.DataFrame({'Unique_id':[uid],'Sentence':[original_line],'category':[k],'Sentiment':[sentiment_score]})           
                temp_result_df=temp_result_df.append(temp_df)
                cat_asgign_ind=1
                a=0 
        elif attribute_column == 'Attribute_And_keyword' and values == 'mv-5':
            cat_asgign_ind = 1
    return temp_result_df,cat_asgign_ind

def keyword_search(uid,line,sentiment_score,k,model_df,keyword_column,key_df,keyword_seq,original_line):
    Keyword_df = model_df
    Keyword_category=Keyword_df['Category']  
    Keyword_separated =Keyword_df[keyword_column].astype(str).str.split(',')
    Keyword_dict = dict(zip(Keyword_category, Keyword_separated))  
    temp_result_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
    temp_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
    a=0
    cat_asgign_ind=0
    if Keyword_dict[k] == ['mv-5']:
        temp_df=key_df
        temp_result_df=temp_result_df.append(temp_df)
        cat_asgign_ind=1
    else:
        for word_temp in Keyword_dict[k]: 
            if cat_asgign_ind == 0:  
                words_separated=word_temp.split(' AND ')
                for word in words_separated:
                    word=word.lower()
                    find=re.compile('\+')   
                    replace='.+'  
                    word = find.sub(replace, word)
                    find=re.compile('\?')   
                    replace='.?'
                    word = find.sub(replace, word)
                    find=re.compile('\"')   
                    replace=''
                    word = find.sub(replace, word)
                    if re.search("\\b%s\\b"%word, line):
                        a=a+1
                if a == len(words_separated):
                    temp_df = pd.DataFrame({'Unique_id':[uid],'Sentence':[original_line],'category':[k],'Sentiment':[sentiment_score]})
                    if keyword_seq==3:            
                        temp_result_df=temp_result_df.append(temp_df)
                    cat_asgign_ind=1
                a=0
    if keyword_seq==3:
        return temp_result_df,cat_asgign_ind
    else:
        return temp_df,cat_asgign_ind
"""Actual core classification function"""
def core_classify(line,uid, sentiment_score,model_df,original_line):     
    temp_result_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
    temp_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
    cat_asgign_ind=0
    cat_asgign_ind_1=0
    cat_asgign_ind_2=0
    attri_or_assign_ind=0
    attri_and_assign_ind=0
    for k in model_df['Category']:        
        if not Exclusion(line,k,model_df):
            temp_df_attribute_or, attri_or_assign_ind=Attribute_check(uid,line,sentiment_score,k,model_df,'Attribute_Or_keyword',original_line)
            temp_df_attribute_and, attri_and_assign_ind=Attribute_check(uid,line,sentiment_score,k,model_df,'Attribute_And_keyword',original_line)
            temp_result_df=temp_result_df.append(temp_df_attribute_or)
            if attri_or_assign_ind == 0 and attri_and_assign_ind == 1:
                temp_df_keywords, cat_asgign_ind=keyword_search(uid,line,sentiment_score,k,model_df,'Keywords',temp_df,1,original_line)
            if cat_asgign_ind == 1:
                temp_df_keywords_and_1, cat_asgign_ind_1=keyword_search(uid,line,sentiment_score,k,model_df,'Keywords_and_1',temp_df_keywords,2,original_line)
                temp_df_keywords_and_2, cat_asgign_ind_2=keyword_search(uid,line,sentiment_score,k,model_df,'Keywords_and_2',temp_df_keywords_and_1,3,original_line)
                if  cat_asgign_ind_1 == 1 and cat_asgign_ind_2 == 1:       
                    temp_result_df=temp_result_df.append(temp_df_keywords_and_2)
    if temp_result_df.empty:
        temp_df = pd.DataFrame({'Unique_id':[uid],'Sentence':[original_line],'category':['No matching category'],'Sentiment':[sentiment_score]})
        temp_result_df=temp_result_df.append(temp_df)
    return temp_result_df

"""For sentiment analysis, to get words followed by negators"""
         
def negators(line):
    lwords = line.split()
    negators=['not','couldnt','didnt','wasnt','isnt','arent','dont','wouldnt','doesnt','cant']
    inverse_sentiment=[]
    for neg in negators:
        indices = [i for i, x in enumerate(lwords) if x == neg]        
        for ind in indices:
            if ind+1 < len(lwords):
                inverse_sentiment.append(lwords[ind+1])
    return inverse_sentiment

"""Function to get the sentiment score of the sentence"""

def sentiment_calc(line):
    sentiments_df = pd.read_csv('sentiment_export.csv')
    Words=sentiments_df['WORD']
    Sentiment_value=sentiments_df['SENTIMENT']
    Sentiment_dict=dict(zip(Words, Sentiment_value))
    score_dict={}    
    for k,v in Sentiment_dict.items():
        k=k.lower()   
        if re.search("\\b%s\\b"%k, line):
            inverse_sentimt_list=negators(line)
            if k in inverse_sentimt_list:
                score_dict[k]=(-1)*v
            else:
                score_dict[k]=v
    sentiment_score = np.mean(score_dict.values())
    return sentiment_score
    
"""To get the frequency of keywords in the data"""
     
def Word_distribution(data_df):
    my_list = data_df["Verbatim"].tolist()
    for p in list(punctuation):
        my_list=str(my_list).replace(p,'')
    my_list = my_list.lower()
    words = nltk.tokenize.word_tokenize(str(my_list))
    words_modified=[]
    exc_words=['not','couldnt','didnt','wasnt','isnt','arent','dont','wouldnt','doesnt','cant','capital','one']
    for i in words:
        if len(i) > 2 and i.lower() not in stopwords.words('english') and i.lower() not in exc_words:
            words_modified.append(i)   
    fdist = FreqDist(words_modified)
    return fdist

"""To get high frequency keywords that are not used in the model"""

def Missing_hf_Keywords(top,fdist,model_df):
    hf_keywords=[]
    hf_frequency={}
    top=50
    for j in [i[0] for i in fdist.most_common(top)]:
       hf_keywords.append(j) 
    model_df.Exclusion=model_df.Exclusion.fillna(-2.0012)
    model_df.Keywords=model_df.Keywords.fillna(-2.0012)
    category=model_df['Category']  
    Exclusion_separated =model_df['Exclusion'].astype(str).str.split(',')
    Exclusion_dict = dict(zip(category, Exclusion_separated))  
    keywords_separated =model_df['Keywords'].astype(str).str.split(',')
    keyword_dict = dict(zip(category, keywords_separated))
    keywords__and_1_separated =model_df['Keywords_and_1'].astype(str).str.split(',')
    keyword_and_1_dict = dict(zip(category, keywords__and_1_separated))
    keywords_and_2_separated =model_df['Keywords_and_2'].astype(str).str.split(',')
    keyword_and_2_dict = dict(zip(category, keywords_and_2_separated))
    def key_search(Dict_data,hf_keywords):
        for k,v in Dict_data.items(): 
            for eword_combine in v:
                eword_separated=eword_combine.split(' AND ')
                for eword in  eword_separated:               
                    for hfword in hf_keywords:
                        eword=eword.lower()
                        eword=eword.strip()
                        find=re.compile('\+')   
                        replace='.+'  
                        eword = find.sub(replace, eword)
                        find=re.compile('\?')   
                        replace='.?'
                        eword = find.sub(replace, eword)
                        find=re.compile('\"')   
                        replace=''
                        eword = find.sub(replace, eword)
                        if re.search("\\b%s\\b"%eword, hfword):
                            hf_keywords.remove(hfword)
        return hf_keywords
    hf_keywords=key_search(keyword_dict,hf_keywords)
    hf_keywords=key_search(keyword_and_1_dict,hf_keywords)
    hf_keywords=key_search(keyword_and_2_dict,hf_keywords)
    hf_keywords=key_search(Exclusion_dict,hf_keywords)
    for i in hf_keywords:
        hf_frequency[i]=fdist[i]
    return hf_frequency    

"""Actual code to call all functions and create the output. Take the survey data, run classification and sentiment analysis"""
"""
model_parsing()
data_df=pd.read_csv('twitter.csv')
data_df.Verbatim=data_df.Verbatim.fillna(0)
unique_id=data_df['Unique_Id']
verbatims=data_df['Verbatim']
data_dict = dict(zip(unique_id, verbatims))
Results_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
model_df = pd.read_csv('Model_modified_twitter_test.csv')
for uid,line in data_dict.items(): 
    line=line.decode('utf-8',errors='ignore') #To make sure program doesnt run into unicode error. Add errot handling to avoid issues with other formats
    try:
        line_list=tokenize.sent_tokenize(str(line))
        for line in line_list:
            original_line=line
            for p in list(punctuation):
                line=line.replace(p,'')
            line=line.lower()
            line_SC=tb.blob.BaseBlob(line)
            line=line_SC.correct()
            line=str(line)
            print uid
            sentiment_score=sentiment_calc(line)
            temp_df=core_classify(line,uid,sentiment_score,model_df,original_line)
            Results_df = Results_df.append(temp_df)
    except UnicodeEncodeError:
        temp_df = pd.DataFrame({'Unique_id':[uid],'Sentence':[original_line],'category':['Invalid text data'],'Sentiment':[sentiment_score]})
        Results_df = Results_df.append(temp_df)
Results_df.to_csv('test_analysis.csv',index=False, encoding = 'utf-8')
"""
"""Get the frequency distribution of the words from data"""
"""
fdist=Word_distribution(data_df)
distribution_df=pd.DataFrame(list(fdist.iteritems()),columns=['Word','Frequency'])
distribution_df=distribution_df.sort(columns='Frequency',ascending=False)
"""
"""Get the frequency distribution of the HF keywords not used in the model"""
"""
hfdist=Missing_hf_Keywords(40,fdist,model_df)
hfwords_df=pd.DataFrame(list(hfdist.iteritems()),columns=['Word','Frequency'])
hfwords_df=hfwords_df.sort(columns='Frequency',ascending=False)
"""
"""TO write output to excel file"""
"""
results = ExcelWriter('Results_data.xlsx')
Results_df.to_excel(results, sheet_name='Claasified_data',index=False)
distribution_df.to_excel(results,sheet_name='Word_Frequency',index=False)
hfwords_df.to_excel(results,sheet_name='HF_Word_Frequency',index=False)
results.save()
"""
def realtime():
    model_parsing()
    data_df=pd.read_csv('Test_Survey.csv')
    data_df.Verbatim=data_df.Verbatim.fillna(0)
    unique_id=data_df['Unique_Id']
    verbatims=data_df['Verbatim']
    data_dict = dict(zip(unique_id, verbatims))
    Results_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
    model_df = pd.read_csv('Model_modified_twitter_test.csv')
    for uid,line in data_dict.items(): 
        line=str(line).decode('utf-8',errors='ignore') #To make sure program doesnt run into unicode error. Add errot handling to avoid issues with other formats
        try:
            line_list=tokenize.sent_tokenize(str(line))
            tokenize.sent_tokenize(str(line))
            for line in line_list:
                original_line=line
                for p in list(punctuation):
                    line=line.replace(p,'')
                line=line.lower()
                line_SC=tb.blob.BaseBlob(line)
                line=line_SC.correct()
                line=str(line)
                #print uid
                sentiment_score=sentiment_calc(line)
                
                temp_df=core_classify(line,uid,sentiment_score,model_df,original_line)
                #Results_df = Results_df.append(temp_df)
                
                yield temp_df
        except UnicodeEncodeError:
            temp_df = pd.DataFrame({'Unique_id':[uid],'Sentence':[original_line],'category':['Invalid text data'],'Sentiment':[sentiment_score]})
            yield temp_df
            #Results_df = Results_df.append(temp_df)
    Results_df.to_csv('test_analysis.csv',index=False, encoding = 'utf-8')

def Word_distribution_line(line):
    my_list = line
    for p in list(punctuation):
        my_list=str(my_list).replace(p,'')
    my_list = my_list.lower()
    words = nltk.tokenize.word_tokenize(str(my_list))
    words_modified=[]
    exc_words=['not','couldnt','didnt','wasnt','isnt','arent','dont','wouldnt','doesnt','cant','capital','one']
    for i in words:
        if len(i) > 2 and i.lower() not in stopwords.words('english') and i.lower() not in exc_words:
            words_modified.append(i)   
    fdist = FreqDist(words_modified)    
    return fdist
    #distribution_df=pd.DataFrame(list(fdist.iteritems()),columns=['Word','Frequency'])
    #return distribution_df



def Missing_hf_Keywords_line(line):
    model_df = pd.read_csv('Model_modified_twitter_test.csv')
    hf_keywords=[]
    hf_frequency={}
    fdist=Word_distribution_line(line)
    top=50
    for j in [i[0] for i in fdist.most_common(top)]:
       hf_keywords.append(j) 
    model_df.Exclusion=model_df.Exclusion.fillna(-2.0012)
    model_df.Keywords=model_df.Keywords.fillna(-2.0012)
    category=model_df['Category']  
    Exclusion_separated =model_df['Exclusion'].astype(str).str.split(',')
    Exclusion_dict = dict(zip(category, Exclusion_separated))  
    keywords_separated =model_df['Keywords'].astype(str).str.split(',')
    keyword_dict = dict(zip(category, keywords_separated))
    keywords__and_1_separated =model_df['Keywords_and_1'].astype(str).str.split(',')
    keyword_and_1_dict = dict(zip(category, keywords__and_1_separated))
    keywords_and_2_separated =model_df['Keywords_and_2'].astype(str).str.split(',')
    keyword_and_2_dict = dict(zip(category, keywords_and_2_separated))
    def key_search(Dict_data,hf_keywords):
        for k,v in Dict_data.items(): 
            for eword_combine in v:
                eword_separated=eword_combine.split(' AND ')
                for eword in  eword_separated:               
                    for hfword in hf_keywords:
                        eword=eword.lower()
                        eword=eword.strip()
                        find=re.compile('\+')   
                        replace='.+'  
                        eword = find.sub(replace, eword)
                        find=re.compile('\?')   
                        replace='.?'
                        eword = find.sub(replace, eword)
                        find=re.compile('\"')   
                        replace=''
                        eword = find.sub(replace, eword)
                        if re.search("\\b%s\\b"%eword, hfword):
                            hf_keywords.remove(hfword)
        return hf_keywords
    hf_keywords=key_search(keyword_dict,hf_keywords)
    hf_keywords=key_search(keyword_and_1_dict,hf_keywords)
    hf_keywords=key_search(keyword_and_2_dict,hf_keywords)
    hf_keywords=key_search(Exclusion_dict,hf_keywords)
    for i in hf_keywords:
        hf_frequency[i]=fdist[i]
    return pd.Series(hf_frequency)
    
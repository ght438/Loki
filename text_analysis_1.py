# -*- coding: utf-8 -*-
"""
Created on Wed Jul 08 14:28:03 2015

@author: GHT438
"""

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
import pyodbc
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
import sys
from nltk.tokenize import word_tokenize
from textblob import Word

def model_parsing():
    Parse_df=model_df.fillna('mv-5')
    for index, row_k  in enumerate(Parse_df['Keywords']):
        row_k=parse_and.parse(str(row_k))
        Parse_df.Keywords[index] = row_k
    for index, row_k  in enumerate(Parse_df['Keywords_and_1']):
        row_k=parse_and.parse(str(row_k))
        Parse_df.Keywords_and_1[index] = row_k
    for index, row_k  in enumerate(Parse_df['Keywords_and_2']):
        row_k=parse_and.parse(str(row_k))
        Parse_df.Keywords_and_2[index] = row_k
    for index, row_e in enumerate(Parse_df['Exclusion']):
        row_e=parse_and.parse(str(row_e))
        Parse_df.Exclusion[index] = row_e
    for index, row_e in enumerate(Parse_df['Attribute_Or_keyword']):
        row_e=parse_and.parse(str(row_e))
        Parse_df.Attribute_Or_keyword[index] = row_e
    for index, row_e in enumerate(Parse_df['Attribute_And_keyword']):
        row_e=parse_and.parse(str(row_e))
        Parse_df.Attribute_And_keyword[index] = row_e
    for index, row_e in enumerate(Parse_df['Verbatim_exclusion']):
        row_e=parse_and.parse(str(row_e))
        Parse_df.Verbatim_exclusion[index] = row_e
    Parse_df.to_csv('Model_modified_twitter_test.csv',index=False)

def spell_check(line):
    modified_line=line
    word_list=word_tokenize(line)
    for word in word_list:
        word=word.lower()
        if word in spell_dict.keys():
            modified_line = re.sub(word,spell_dict[word],line)
        elif word.isalnum():
            search = open('English_words.txt', 'r')
            if word not in english_dict and word not in search.read():
                w = Word(word)
                suggestion=w.spellcheck()
                if max(suggestion)[1] > 0.9:          
                    word_checked=max(suggestion)[0]   
                    spell_dict[word]=word_checked
                    modified_line = re.sub(word,spell_dict[word],modified_line)
    return modified_line
    
"""Function to Exclude a category based on pre-defined keywords criteria"""
def Exclusion(line,k):
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
            find=re.compile('\*')   
            replace='.+'  
            eword = find.sub(replace,eword)
            find=re.compile('\?')   
            replace='.?'
            eword = find.sub(replace,eword)
            find=re.compile('\"')   
            replace=''
            eword = find.sub(replace,eword)
            if  re.search("\\b%s\\b"%eword.strip(), line):
                a=a+1
        if a == len(words_separated):
            exc_flag=1
        a=0
    if exc_flag==1:
        return True
    else:
        return False
        
def verbatim_exclusion(line):
    for p in list(punctuation):
        line=line.replace(p,'')
    line=line.lower()
    line=spell_check(line)
    Exclusion_verbatim_df = model_df
    Exclusion_verbatim_df.Verbatim_exclusion=Exclusion_verbatim_df.Verbatim_exclusion.fillna(0)
    Exclusion__verbatim_category=Exclusion_verbatim_df['Category']  
    Exclusion__verbatim_separated =Exclusion_verbatim_df['Verbatim_exclusion'].astype(str).str.split(',')
    Exclusion_verbatim_dict = dict(zip(Exclusion__verbatim_category, Exclusion__verbatim_separated))  
    Verbatim_exc_list=[]    
    a=0
    for k,v in Exclusion_verbatim_dict.items():
        for eword_temp in v: 
            words_separated=eword_temp.split(' AND ')
            for eword in words_separated:
                eword=eword.lower()
                find=re.compile('\+')   
                replace='.+'  
                eword = find.sub(replace,eword)
                find=re.compile('\*')   
                replace='.+'  
                eword = find.sub(replace,eword)
                find=re.compile('\?')   
                replace='.?'
                eword = find.sub(replace,eword)
                find=re.compile('\"')   
                replace=''
                eword = find.sub(replace,eword)
                if  re.search("\\b%s\\b"%eword.strip(), line):
                    a=a+1
            if a == len(words_separated):
                Verbatim_exc_list.append(k)
            a=0
    return Verbatim_exc_list 
        
def Attribute_check(uid,line,sentiment_score,k,attribute_column,original_line):
    Attribute_df = model_df
    Attribute_category=Attribute_df['Category']  
    Attrbute_separated =Attribute_df[attribute_column].astype(str).str.split(',')
    Attribute_dict = dict(zip(Attribute_category, Attrbute_separated))
    temp_result_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
    att_data_df=data_df 
    att_data_df=att_data_df.fillna('mv-a2')
    cat_asgign_ind=0
    a=0
    for values in Attribute_dict[k]:
        if cat_asgign_ind == 0 and values != 'mv-5':
            values_separated=values.split(' AND ')
            for value in values_separated:
                fields = re.split(r'(==|>|<|!=|>=|<=)\s*', value)
                try:     
                    attribute_name=fields[0].strip()
                    attribute_actual= att_data_df['%s'%attribute_name].loc[att_data_df['Unique_id'] == uid].values
                except KeyError:
                    print attribute_name
                    #sys.exit("Specified attribute, %s , from model does not exist in the datafile, Please check and rerun the classification"%attribute_name)
                    print "Specified attribute does not exist in the datafile, Please check and rerun the classification"                
                if attribute_actual[0] != 'mv-a2':
                    try:
                        ops = {'==' : operator.eq,'!=' : operator.ne,'<=' : operator.le,'>=' : operator.ge,'>'  : operator.gt,'<'  : operator.lt}
                        try:
                            if ops[fields[1]](float(attribute_actual[0]),float(fields[2])):
                                a=a+1
                        except:
                            find=re.compile('\'|\"')   
                            replace=''
                            check_value = find.sub(replace, fields[2])
                            actual_attribute = find.sub(replace, attribute_actual[0])
                            if ops['=='](str(check_value).lower(), str(actual_attribute).lower()):                              
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

def keyword_search(uid,line,sentiment_score,k,keyword_column,key_df,keyword_seq,original_line):
    Keyword_df = model_df
    Keyword_category=Keyword_df['Category']  
    Keyword_separated =Keyword_df[keyword_column].astype(str).str.split(',')
    Keyword_dict = dict(zip(Keyword_category, Keyword_separated))  
    temp_result_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
    temp_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
    a=0
    cat_asgign_ind=0
    if Keyword_dict[k] == ['mv-5'] and keyword_seq not in (1,4):
        temp_df=key_df
        temp_result_df=temp_result_df.append(temp_df)
        cat_asgign_ind=1
    elif Keyword_dict[k] == ['mv-5'] and keyword_seq == 4:
        temp_df=key_df
        temp_result_df=temp_result_df.append(temp_df)
        cat_asgign_ind=0
    elif Keyword_dict[k] == ['mv-5'] and keyword_seq == 1:
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
                    find=re.compile('\*')   
                    replace='.+'  
                    word = find.sub(replace, word)
                    find=re.compile('\?')   
                    replace='.?'
                    word = find.sub(replace, word)
                    find=re.compile('\"')   
                    replace=''
                    word = find.sub(replace, word)
                    if re.search("\\b%s\\b"%word.strip(), line):
                        a=a+1
                if a == len(words_separated):
                    temp_df = pd.DataFrame({'Unique_id':[uid],'Sentence':[original_line],'category':[k],'Sentiment':[sentiment_score]})
                    if keyword_seq in (3,6):            
                        temp_result_df=temp_result_df.append(temp_df)
                    cat_asgign_ind=1
                a=0
    if keyword_seq in (3,6):
        return temp_result_df,cat_asgign_ind
    else:
        return temp_df,cat_asgign_ind

def verbatim_core_classify(line,uid, sentiment_score,original_line,Verbatim_exc_list):     
    temp_result_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
    temp_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
    cat_asgign_ind=0
    cat_asgign_ind_1=0
    cat_asgign_ind_2=0
    attri_or_assign_ind=0
    attri_and_assign_ind=0
    Verbatim_inc_list=[]
    for p in list(punctuation):
        line=line.replace(p,'')
    line=line.lower()
    line=spell_check(line)
    for k in model_df['Category']:    
        temp_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))  
        temp_df_keywords=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
        temp_df_keywords_and_1=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
        temp_df_keywords_and_2=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
        if k not in Verbatim_exc_list:
            temp_df_attribute_or, attri_or_assign_ind=Attribute_check(uid,line,sentiment_score,k,'Attribute_Or_keyword',original_line)
            temp_df_attribute_and, attri_and_assign_ind=Attribute_check(uid,line,sentiment_score,k,'Attribute_And_keyword',original_line)
            temp_result_df=temp_result_df.append(temp_df_attribute_or)
            if attri_or_assign_ind == 0 and attri_and_assign_ind == 1:
                temp_df_keywords, cat_asgign_ind=keyword_search(uid,line,sentiment_score,k,'Keywords_verb',temp_df,4,original_line)
            if cat_asgign_ind == 1:
                temp_df_keywords_and_1, cat_asgign_ind_1=keyword_search(uid,line,sentiment_score,k,'Keywords_verb_and_1',temp_df_keywords,5,original_line)
                temp_df_keywords_and_2, cat_asgign_ind_2=keyword_search(uid,line,sentiment_score,k,'Keywords_verb_and_2',temp_df_keywords_and_1,6,original_line)
                if  cat_asgign_ind_1 == 1 and cat_asgign_ind_2 == 1:       
                    temp_result_df=temp_result_df.append(temp_df_keywords_and_2)
                    if not temp_result_df.empty:      
                        Verbatim_inc_list.append(k) 
    return Verbatim_inc_list

"""Actual core classification function"""
def core_classify(line,uid, sentiment_score,original_line,Verbatim_exc_list,Verbatim_inc_list):     
    temp_result_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
    temp_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
    cat_asgign_ind=0
    cat_asgign_ind_1=0
    cat_asgign_ind_2=0
    attri_or_assign_ind=0
    attri_and_assign_ind=0
    for k in model_df['Category']:
        temp_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment')) 
        temp_df_attribute_or=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
        temp_df_attribute_and=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
        temp_df_keywords=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
        temp_df_keywords_and_1=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
        temp_df_keywords_and_2=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
        if not Exclusion(line,k) and k not in Verbatim_exc_list and k not in Verbatim_inc_list:
            temp_df_attribute_or, attri_or_assign_ind=Attribute_check(uid,line,sentiment_score,k,'Attribute_Or_keyword',original_line)
            temp_df_attribute_and, attri_and_assign_ind=Attribute_check(uid,line,sentiment_score,k,'Attribute_And_keyword',original_line)
            temp_result_df=temp_result_df.append(temp_df_attribute_or)
            if attri_or_assign_ind == 0 and attri_and_assign_ind == 1:
                temp_df_keywords, cat_asgign_ind=keyword_search(uid,line,sentiment_score,k,'Keywords',temp_df_attribute_and,1,original_line)
            if cat_asgign_ind == 1:
                temp_df_keywords_and_1, cat_asgign_ind_1=keyword_search(uid,line,sentiment_score,k,'Keywords_and_1',temp_df_keywords,2,original_line)
                temp_df_keywords_and_2, cat_asgign_ind_2=keyword_search(uid,line,sentiment_score,k,'Keywords_and_2',temp_df_keywords_and_1,3,original_line)            
                if  cat_asgign_ind_1 == 1 and cat_asgign_ind_2 == 1:       
                    temp_result_df=temp_result_df.append(temp_df_keywords_and_2)
    if temp_result_df.empty and not Verbatim_inc_list:
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
     
def Word_distribution():
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

def Missing_hf_Keywords(top,fdist):
    HF_Key_df=model_df
    hf_keywords=[]
    hf_frequency={}
    top=50
    for j in [i[0] for i in fdist.most_common(top)]:
       hf_keywords.append(j) 
    HF_Key_df.Exclusion=HF_Key_df.Exclusion.fillna(-2.0012)
    HF_Key_df.Keywords=HF_Key_df.Keywords.fillna(-2.0012)
    category=HF_Key_df['Category']  
    Exclusion_separated =HF_Key_df['Exclusion'].astype(str).str.split(',')
    Exclusion_dict = dict(zip(category, Exclusion_separated))  
    keywords_separated =HF_Key_df['Keywords'].astype(str).str.split(',')
    keyword_dict = dict(zip(category, keywords_separated))
    keywords__and_1_separated =HF_Key_df['Keywords_and_1'].astype(str).str.split(',')
    keyword_and_1_dict = dict(zip(category, keywords__and_1_separated))
    keywords_and_2_separated =HF_Key_df['Keywords_and_2'].astype(str).str.split(',')
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
                        find=re.compile('\*')   
                        replace='.+'  
                        eword = find.sub(replace, eword)
                        find=re.compile('\?')   
                        replace='.?'
                        eword = find.sub(replace, eword)
                        find=re.compile('\"')   
                        replace=''
                        eword = find.sub(replace, eword)
                        if re.search("\\b%s\\b"%eword.strip(), hfword):
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
global model_df
global data_df
global spell_dict
global english_dict
conn = pyodbc.connect('DRIVER={Teradata};DBCNAME=oneview;UID=ght438;PWD=Navya345;',autocommit=True)
data_df = pd.read_sql("""SEL 
      cmplant_id AS Unique_id,
      'T3_USCARD_COMPLAINT' AS study_type, 
      date_received AS cmplant_case_recvd_dt, 
      cmplant_cmnt_txt AS Verbatim,
      primary_driver AS PRIM_CMPLANT_DRVR_NM,
      primary_subdriver AS PRIM_CMPLANT_SUB_DRVR
FROM ud401.cmplant_pl
WHERE 
      data_source='Chordiant'
      AND date_received BETWEEN DATE'2015-06-01' AND '2015-06-30';""", conn)
spell_dict={}
english_dict=nltk.corpus.words.words()
model_df=pd.read_csv('Test_complaints.csv')
"""Use this part to get input from the excel file
try:
    data_df=pd.read_csv('T3_Complaint_Data_V1.csv')
except:
    reload(sys)
    sys.setdefaultencoding('Latin-1')
    data_df=pd.read_csv('T3_Complaint_Data_V1.csv')
"""
model_parsing()
data_df.Verbatim=data_df.Verbatim.fillna(0)
Unique_id=data_df['Unique_id']
verbatims=data_df['Verbatim']
data_dict = dict(zip(Unique_id, verbatims))
Results_df=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
model_df = pd.read_csv('Model_modified_twitter_test.csv')
for uid,line in data_dict.items(): 
    try:
        reload(sys)
        sys.setdefaultencoding('utf-8')
        line=str(line).decode('utf-8') #To make sure program doesnt run into unicode error. Add errot handling to avoid issues with other formats
    except:
        reload(sys)
        sys.setdefaultencoding('Latin-1')
        line=str(line).decode('Latin-1') #To make sure program doesnt run into unicode error. Add errot handling to avoid issues with other formats
    original_verbatim=line 
    line=spell_check(line)
    Verbatim_exc_list=verbatim_exclusion(line)    
    sentiment_score_verbatim=sentiment_calc(line)
    Verbatim_inc_list=verbatim_core_classify(line,uid,sentiment_score_verbatim,original_verbatim,Verbatim_exc_list)     
    try:
        line_list=tokenize.sent_tokenize(str(line))
        for line in line_list:
            inter_df_verbatim=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
            original_line=line
            for p in list(punctuation):
                line=line.replace(p,'')
            line=line.lower()
            print "Processing ID # %s"%uid 
            sentiment_score=sentiment_calc(line)
            if Verbatim_inc_list:
                for category in Verbatim_inc_list:
                    temp_df_verbatim = pd.DataFrame({'Unique_id':[uid],'Sentence':[original_line],'category':[category],'Sentiment':[sentiment_score]})
                    inter_df_verbatim = inter_df_verbatim.append(temp_df_verbatim)
            temp_df=core_classify(line,uid,sentiment_score,original_line,Verbatim_exc_list,Verbatim_inc_list)
            Results_df = Results_df.append(temp_df)
            Results_df = Results_df.append(inter_df_verbatim)
            inter_df_verbatim=pd.DataFrame(columns=('Unique_id','Sentence', 'category', 'Sentiment'))
    except UnicodeEncodeError:
        temp_df = pd.DataFrame({'Unique_id':[uid],'Sentence':["Invalid text"],'category':['Invalid text data'],'Sentiment':[sentiment_score]})
        Results_df = Results_df.append(temp_df)
try:
    reload(sys)  # Reload does the trick!
    sys.setdefaultencoding('utf-8')
    Results_df.to_csv('test_analysis.csv',index=False)
except:
    reload(sys)  # Reload does the trick!
    sys.setdefaultencoding('Latin-1')
    Results_df.to_csv('test_analysis.csv',index=False)
final_result_df=pd.merge(data_df, Results_df, how='inner', on='Unique_id')
#data_df.to_csv('data_temp.csv',index=False,encoding='utf-8')
"""Get the frequency distribution of the words from data"""
fdist=Word_distribution()
distribution_df=pd.DataFrame(list(fdist.iteritems()),columns=['Word','Frequency'])
distribution_df=distribution_df.sort(columns='Frequency',ascending=False)

"""Get the frequency distribution of the HF keywords not used in the model"""
hfdist=Missing_hf_Keywords(40,fdist)
hfwords_df=pd.DataFrame(list(hfdist.iteritems()),columns=['Word','Frequency'])
hfwords_df=hfwords_df.sort(columns='Frequency',ascending=False)
"""TO write output to excel file"""

try:
    reload(sys)  # Reload does the trick!
    sys.setdefaultencoding('utf-8')
    results = ExcelWriter('T3_complaints_output.xlsx')
    final_result_df.to_excel(results, sheet_name='Claasified_data',index=False)
    distribution_df.to_excel(results,sheet_name='Word_Frequency',index=False)
    hfwords_df.to_excel(results,sheet_name='HF_Word_Frequency',index=False)
    results.save()
except:
    reload(sys)  # Reload does the trick!
    sys.setdefaultencoding('Latin-1')
    results = ExcelWriter('T3_complaints_output.xlsx')
    final_result_df.to_excel(results, sheet_name='Claasified_data',index=False)
    distribution_df.to_excel(results,sheet_name='Word_Frequency',index=False,encoding='Latin-1')
    hfwords_df.to_excel(results,sheet_name='HF_Word_Frequency',index=False,encoding='Latin-1')
    results.save()





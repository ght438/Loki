# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 09:39:04 2015

@author: BXZ747
"""

import Tkinter as tk
from ttk import Treeview, Style
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import Text_analytics_gui

#next value
next=Text_analytics_gui.realtime().next

#Import data
classifiedData=next()
classifiedData['Sentiment']=classifiedData['Sentiment'].astype(float)
topWords=Text_analytics_gui.Missing_hf_Keywords_line(classifiedData['Sentence'])
topWords.sort()

#Category counts
categories=classifiedData["category"][classifiedData["category"] != "No matching category"].value_counts()   
categories.sort(ascending=False)
catData=categories[:5]

#Sentiment
sentiment=classifiedData[classifiedData["category"] != "No matching category"].pivot_table(values='Sentiment', index='category', aggfunc=np.mean)
sentiment.fillna(0) #fix NaN
sentData=sentiment[catData.index]

#sample
sample=classifiedData[["Sentence","category"]][classifiedData["category"] != "No matching category"]

#Create the GUI
root=tk.Tk()
root.wm_title("Text Analytics")
mainPan=tk.PanedWindow(root,orient=tk.HORIZONTAL)
leftPan=tk.PanedWindow(mainPan,orient=tk.VERTICAL)
#midPan=tk.PanedWindow(mainPan, orient=tk.VERTICAL)
rightPan=tk.PanedWindow(mainPan,orient=tk.VERTICAL)
mainPan.add(leftPan)
#mainPan.add(midPan)
mainPan.add(rightPan)

#Top categories
lab1=tk.Label(leftPan,text="Top Categories", font='arial 10 bold')
leftPan.add(lab1)
lab1.pack()

catFig=plt.Figure(figsize=(8, 6))
catSubPlt=catFig.add_subplot(111)

catBar=catSubPlt.barh(np.arange(5), [catData.values[0],0,0,0,0],align='center')
catSubPlt.set_yticks(np.arange(5))
catSubPlt.set_yticklabels([catData.index[0],' ',' ',' ',' '])
topLeft = FigureCanvasTkAgg(catFig, master=leftPan)
topLeft.show()
topLeft.get_tk_widget().pack()
catFig.tight_layout()

#Sentiment for top categories
lab2=tk.Label(leftPan,text="Sentiment Score",font='arial 10 bold')
leftPan.add(lab2)
lab2.pack()

sentFig=plt.Figure(figsize=(8,4))
sentSubPlt=sentFig.add_subplot(111)
sentBar=sentSubPlt.barh(np.arange(5), [sentData.values[0],0,0,0,0], align='center',color='green')
sentSubPlt.set_yticks(np.arange(5))
sentSubPlt.set_yticklabels([sentData.index[0],' ',' ',' ',' '])
bottomLeft = FigureCanvasTkAgg(sentFig, master=leftPan)
bottomLeft.show()
bottomLeft.get_tk_widget().pack()
sentFig.tight_layout()

#Top Keywords
lab3=tk.Label(rightPan,text="Most Frequent Unmodeled Keywords",font='arial 10 bold')
rightPan.add(lab3)
lab3.pack()

keywords=Treeview(rightPan)
keywords["columns"]=["Frequency"]
style = Style(root)
style.configure('Treeview', rowheight=29,font='arial 10 bold')

keywords.heading("#0", text="Word")
keywords.heading("Frequency", text="Frequency")
keywords.column("#0", anchor=tk.CENTER)
keywords.column("Frequency", anchor=tk.CENTER)
keywordIndex={}
top10Words=topWords[-10:]

tree_i=0
for index,row in top10Words.iteritems():
    treeind=keywords.insert(parent="",index=0,text=str(index), values=(str(row)))
    keywordIndex[tree_i]=treeind
    tree_i+=1
    
rightPan.add(keywords)
keywords.pack()


#Sample    
lab4 =tk.Label(rightPan, text="Categorization Sample",font='arial 10 bold')
rightPan.add(lab4)
lab4.pack()


scrollbar = tk.Scrollbar(rightPan)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

listbox = tk.Text(rightPan,wrap=tk.WORD,width=55,height=40)
listbox.pack()

# attach listbox to scrollbar
listbox.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=listbox.yview)
for index,row in sample.iterrows():
    txt=str(row['Sentence']).strip()
    cat=str(row['category']).strip()
    listbox.insert('1.0', cat+":\n"+txt+"\n\n")
    listbox.config(font="arial 10 bold")

#Pack
leftPan.pack(side=tk.LEFT,fill=tk.BOTH)
rightPan.pack(side=tk.RIGHT,fill=tk.BOTH)
mainPan.pack(fill=tk.BOTH)

#Update function
def update_all():
    global categories, sentiment
    #get next
    nxt= next()
    for i,r in nxt.iterrows():
        categoryName=r["category"]
        sentimentScore=r["Sentiment"]
        sentence=r["Sentence"]
        
        #update word counts
        word_dist=Text_analytics_gui.Missing_hf_Keywords_line(sentence)
        for wrd,addFreq in word_dist.iteritems():            
            if wrd in topWords.keys():
                newFreq=topWords[wrd]+addFreq
            else:
                newFreq=addFreq
            topWords[wrd]=newFreq
        topWords.sort(ascending=True)
        top10Words=topWords[-10:]  
        tree_i=0
        for wrd, freq in top10Words.iteritems():
            treeind=keywords.insert(parent="",index=0,text=str(wrd), values=str(freq))
            keywordIndex[tree_i]=treeind
            tree_i+=1
        
        if categoryName !="No matching category":
            if categoryName not in categories.keys():
                categories[categoryName]=0
                sentiment[categoryName]=0            
            
            #Update sentiment score
            if not np.isnan(sentimentScore):
                sentiment[categoryName]=(categories[categoryName]*sentiment[categoryName]+sentimentScore)/(categories[categoryName]+1)
            #Update top categories
            categories[categoryName]+=1
            
            #Update category bar chart   
            categories.sort(ascending=False)
            catData=categories[:5]
            
            for rec, h in zip(catBar, catData.values):
                rec.set_width(h)
            catSubPlt.set_yticklabels(catData.index)
            catFig.tight_layout()
            #Rescale
            catSubPlt.relim()
            catSubPlt.autoscale_view(True,True,True)
                               
            #Update sentiment chart
            sentData=sentiment[catData.index]
            
            sentDatavs=sentData.values
            sentDatais=sentData.index
            
            while len(sentDatavs)<5:
                sentDatavs=np.append(sentDatavs,0)
                
                
            for rec, h in zip(sentBar, sentDatavs):
                rec.set_width(h)
                if h<0:
                    rec.set_color('red')
                else:
                    rec.set_color('green')
            
            
            sentSubPlt.set_yticks(np.arange(5))
            sentSubPlt.set_yticklabels(sentDatais)
            
            #Rescale
            sentFig.tight_layout()
            sentSubPlt.relim()
            sentSubPlt.autoscale_view(True,True,True)
            topLeft.draw()
            bottomLeft.draw()
        
            #Update sample
            listbox.insert('1.0', categoryName+":\n"+sentence+"\n\n")
        
    root.after(2500, update_all)
    
#gen=test.iterrows()


root.after(2500, update_all)
root.mainloop()



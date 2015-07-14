# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:32:51 2015

Based on parsing info from http://effbot.org/zone/simple-top-down-parsing.htm

@author: bxz747
"""
import re
token=''
    
"""
Classes for the different of tokens

LBP: Left binding power. Higher LBPs take precedence in order of operations
NUD: Operator function for operators whose appear at the beginning of the operation
LED: Operator function for operators whose appear in the middle of the operation
"""

#A string value
class string_token:
    id="str"
    def __init__(self, value):
        self.value = str(value)
    def nud(self):
        return self.value

#AND operator
class and_token:
    id="AND"
    lbp = 20
    def led(self, left):
        #Need to be lists
        right = expression(20)
        if not isinstance(right, list):
            right=[right]
        if not isinstance(left, list):
            left=[left]
        l=[]
        for a in right:
            for b in left:
                l.append(str(b) + " AND " + str(a))
        return l

#Left parenthesis
class left_paren_token:
    id="("
    lbp= 0
    def nud(self):
        expr = expression()
        advance(")")
        return expr
    
#Right paren
class right_paren_token:
    id=")"
    lbp=0
        
#Comma
class comma_token:
    id=","
    lbp=10
    def led(self, left):
        right=expression(10)
        if right==None:
            right=[]
        if not isinstance(left, list):
            left=[left]
        if not isinstance(right, list):
            right=[right]
        for a in right:
            left.append(a)
        
        return left
        
#End token (denotes end of parsing)
class end_token:
    id="END"
    lbp = 0

#Helper fucntion. Evalulate expression up until next given char
def advance(id=None):
    global token
    if id and token.id != id:
        raise SyntaxError("Expected %r" % id)
    token = next()


"""
Methods to tokenize a string containing categorization rulse
"""

#Given a string expression, generate a list of tokens
def genTokens(expr):
    lst=[expr]
    toklist=[" AND ", ",", "\(" , "\)" ] #tokens to check for (escape special chars)
    for tok in toklist:
        tokenized=[]
        for txt in lst:
            tokenized= tokenized + re.split('(%s)' %tok, txt)
        lst=tokenized
    return lst
        
#Tokenize a string containing categoriztaion rules
def tokenize(rules):
    for val in genTokens(rules):
        val=val.strip()
        if val != '': #ignore empty matches
            if val == "(":
                yield left_paren_token()
            elif val == ")":
                yield right_paren_token()
            elif val == "AND":
                yield and_token()
            elif val == ",":
                yield comma_token()
            else:        
                yield string_token(val)
    yield end_token()

"""
Methods to parse a string of rules and evaluate the expression
"""    
#Helpfer function. Returns a string from list or list of lists. Separated by commas
def listString(lst):
    if not isinstance(lst, list):
        return lst
    txt=lst[0]
    for a in lst[1:]:
        txt=txt+", "+listString(a)
    return txt
    
#Evaluate a series of tokens
def expression(rbp=0):
    global token
    t = token
    if isinstance(token, end_token):
       return None 
    #print t.id
    token = next()
    left = t.nud()
    #print str(left)
    while rbp < token.lbp:
        t = token
        token = next()
        left = t.led(left)
    return left

#Main method
#Parse a string of rules and return a string containing the evaluated expression
def parse(rule):
    global token, next
    next = tokenize(rule).next
    token = next()
    return listString(expression())

#test="CLIP, ((increas*, higher) AND (limit, line, credit))"
#test2="(never AND (*enroll*, authorize*, agree*)), not authorize, realize*, (disput* AND enroll*)"
#test3="(increase*, higher) AND limit"
#test4="Protection Bureau, ((world* AND elite))"
#l="(((cash reward, reward points, points, fulfillment, program, mileage, reward+, incentive, rewards program, miles, rewards, reward, cashback, cash back, miles, cash rewards, cash reward, points) AND (reward+, award+, points, redeem, redemption, miles, rewards checking, reward checking, rewards/incentive, reward credit, reward credits, bonus, bonuses, deal, deals, special offer, special offers, special interest, special rate, special rates, mileage, system, membership, qualifi+, deal with, big deal, great, like, hate, disappointed, earn, get, accumulated points, desirable, happy, unhappy, mad, upset, love, hate, low, best, worst, competitive, attractive, unattractive, pitiful)) AND (capital one,credit card,card,cap1,((quicksilver, quicksilverone, quicksilver one,venture, ventureone) AND (card,capital one,capone,cap1,capitalone,credit card))))"
#print parse(l)
#l="a,b,"
#print parse(test2)
#print parse(test3)
#print parse(test)
#print parse(l)
#print parse("a,b,c,")
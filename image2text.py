#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (Madhav Jariwala / makejari)
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25

def calculate_emission_probabilities(train_letters,test_letters):#Calculating emission probabilities
    emission_probabilities = {}
    emission_count = {}
    for letter in range(len(test_letters)):
        emission_probabilities[letter] = {}#Creating a empty dictionary inside a letter key such as {0:{'A':0.343,'B':0.324,....},1:{'A':0.343,'B':0.324,...}}} where 0,1 are are index of the test letters
        emission_count[letter] = {}
        for train in train_letters:#we are checking the emission probability for all possible characters mentioned in the TRAIN_LETTER variable
            number_matching_pixels_pixels = len([(train_letters[train][x][y],test_letters[letter][x][y]) for x in range(CHARACTER_HEIGHT) \
                            for y in range(CHARACTER_WIDTH) if (train_letters[train][x][y]==test_letters[letter][x][y] and test_letters[letter][x][y]=='*')])#check how many pixels match where pixel is '*'
            number_of_empty_pixels = len([(train_letters[train][x][y],test_letters[letter][x][y]) for x in range(CHARACTER_HEIGHT) \
                            for y in range(CHARACTER_WIDTH) if (train_letters[train][x][y]==test_letters[letter][x][y] and test_letters[letter][x][y]==' ')])#check how many pixels match where pixel is ' '(empty)
            emission_cost = (number_matching_pixels_pixels) * 0.9 + (number_of_empty_pixels) * 0.1#Assigning these values to get better result, derived these values after trying a myriad of combinations
            emission_count[letter][train] = emission_cost
    for letter in range(len(test_letters)):
        for train in train_letters:
            # emission_probabilities[letter][train] = (emission_count[letter][train] + 1) / (sum(emission_count[letter].values())+ len(train_letters) * 2) #referenced from https://www.analyticsvidhya.com/blog/2021/04/improve-naive-bayes-text-classifier-using-laplace-smoothing/
            emission_probabilities[letter][train] = (emission_count[letter][train] + 1) / (sum(emission_count[letter].values())+ 2)#smoothing the probability
    return emission_probabilities


def calculate_initial_probabilities(data,total_letter_list):
    initial_count = {}
    initial_probabilities = {}
    for letter in total_letter_list:#this dictionary would look like {'A':0.343,'B':034343,.....,' ':0.00122}
        initial_count[letter] = 0
    for word in range(len(data)):#this is the sentence list
        if data[word][0] in total_letter_list:#checking if the first letter of the sentence is inside the given TRAIN_LETTERS list or not
            initial_count[data[word][0]] += 1
    for dict_letter in initial_count:
        initial_probabilities[dict_letter] = (initial_count[dict_letter] + 1) / (sum(initial_count.values()) + 2)#calcualting the probability
    return initial_probabilities


def calculate_transition_probabilities_dictionary(data,total_letter_list):
    added_string = ' '.join([word for word in data])#Joining all the words list so that we could use it for training
    transition_count = {}
    transition_probabilities = {}
    for letter in total_letter_list:#iterate over TRAIN_LETTERS
        transition_count[letter] = {}#add a key in disctionary - {'A':{},'B':{},....}
        transition_probabilities[letter] = {}
        for next_letter in total_letter_list:
            transition_count[letter][next_letter] = 0# {'A':{'A':0,'B':0,.....},'B':{'A':0,'B':0}}
            transition_probabilities[letter][next_letter] = -np.inf#set the value as infinity
    for letter in range(1,len(added_string)):
        if added_string[letter] in total_letter_list and added_string[letter-1] in total_letter_list:
            transition_count[added_string[letter-1]][added_string[letter]] += 1#calculating the count
    for i in range(len(total_letter_list)):
        for j in range(len(total_letter_list)):
            transition_probabilities[total_letter_list[i]][total_letter_list[j]] = (transition_count[total_letter_list[i]][total_letter_list[j]]+1)/(sum(transition_count[total_letter_list[i]].values())+2)#calculating the probability
        
    return transition_probabilities


    
def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)

    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

def read_data(fname):
    exemplars = []
    file = open(fname, 'r')
    for line in file:
        data = tuple([w for w in line.split()])
        exemplars += data[0::2]
    return exemplars

def calculate_using_bayes(test_letters,train_letters):    
    emission_probabilities = calculate_emission_probabilities(train_letters,test_letters)
    bayes_string = ''
    for letter in emission_probabilities:
        max_key = max(emission_probabilities[letter], key=emission_probabilities[letter].get)
        bayes_string += max_key
    return bayes_string
    


def calculated_using_viterbi(data,train_word_list,test_letters,train_letters,lines):
    emission_probabilities = calculate_emission_probabilities(train_letters,test_letters)
    initial_probabilities = calculate_initial_probabilities(lines,train_word_list)
    transition_probabilities = calculate_transition_probabilities_dictionary(data, train_word_list)    
    
    storing_probabilities = np.ones((len(train_word_list),len(test_letters)))#create a matrix with rows representing hidden states of given observed state
    storing_letter = np.ones((len(train_word_list),len(test_letters)))#creating the matrix that stores max of indexes of previous observed state
    for letter in range(len(test_letters)):#for each letter in test_letter we want to predict    
        for train in range(len(train_word_list)):#compute over all hidden states at given observed state 
            if letter==0:#if it is the first letter we only compute emission and initial probability
                storing_probabilities[train][letter] =  np.log(emission_probabilities[letter][train_word_list[train]]) + np.log(initial_probabilities[train_word_list[train]]) * 0.01
                storing_letter[train][letter] = train#current hidden state value is considered max
            
            else:
                storing_probabilities[train][letter] = np.max([storing_probabilities[t][letter-1]
                        +np.log(transition_probabilities[train_word_list[t]][train_word_list[train]])*0.01
                        +np.log(emission_probabilities[letter][train_word_list[train]]) for t in range(len(train_word_list))])#store the maximum value
                storing_letter[train][letter] = np.argmax([storing_probabilities[t][letter-1]
                        +np.log(transition_probabilities[train_word_list[t]][train_word_list[train]])*0.01
                        +np.log(emission_probabilities[letter][train_word_list[train]]) for t in range(len(train_word_list))])#store the index of maximum value we get from previous hidden state

    best_pointer =np.argmax([storing_probabilities[t][len(test_letters)-1] for t in range(len(train_word_list))])#Last observed state maximum value and its index to backtrack
    backtrack = []
    temp_best_pointer = best_pointer
    for back_pos in range(len(test_letters),0,-1):
        temp = train_word_list[temp_best_pointer]
        backtrack.append(temp)
        temp_best_pointer = int(storing_letter[temp_best_pointer][back_pos-1])   
    
        
    return ''.join(backtrack[::-1])#reversing the backtracked list


#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
data = read_data(train_txt_fname)

f = open(train_txt_fname,'r')
lines = f.readlines()


bayes = calculate_using_bayes(test_letters,train_letters)


viterbi = calculated_using_viterbi(data,TRAIN_LETTERS,test_letters,train_letters,lines)


# The final two lines of your output should look something like this:
print("Simple: " + bayes)
print("   HMM: " + viterbi) 


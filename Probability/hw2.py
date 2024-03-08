import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment


    # Codes below are from chatgpt cause I've never learned how to load the data from a csv file or other files like txt
    X = {chr(i): 0 for i in range(65, 91)}

    with open(filename, encoding='utf-8') as f:
        for line in f:
            for char in line:
                if char.upper() in X:
                    X[char.upper()] += 1
    #Chatgpt citation ends here

    return X



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
def compute(X, e, s):
    # Q1: Print character counts
    print("Q1")
    for char, count in sorted(X.items()):  # Sorting ensures A-Z order
        print(f"{char} {count}")

    probability_english = 0.6  # the probability of English
    probability_spanish = 0.4  # the probability of Spanish

    # Q2: Specific computation for the letter 'A'
    print("Q2")
    print('{:.4f}'.format(X.get('A') * math.log(e[0])))
    print('{:.4f}'.format(X.get('A') * math.log(s[0])))

    # Q3: Log probability of the entire observed count vector X given the language
    # here is where log_prob_english/spanish come from: P(Y =y|X)= f(y) / (f(English) + f(Spanish))
    total_prob_english = sum([X.get(chr(i + 65)) * math.log(e[i]) for i in range(26)])
    total_prob_spanish = sum([X.get(chr(i + 65)) * math.log(s[i]) for i in range(26)])
    log_prob_english = math.log(probability_english) + total_prob_english
    log_prob_spanish = math.log(probability_spanish) + total_prob_spanish
    print("Q3")
    print('{:.4f}'.format(log_prob_english))
    print('{:.4f}'.format(log_prob_spanish))

    # Q4: Conditional probability P(Y = English | X)
    # This part is from the end of 1.3 from pdf
    print("Q4")
    if ((log_prob_spanish - log_prob_english) >= 100):
        print(0)
    elif((log_prob_spanish - log_prob_english) <= -100):
        print(1)
    else:
        print('{:.4f}'.format(1 / (1 + math.exp(log_prob_spanish - log_prob_english))))

if __name__ == "__main__":
    e, s = get_parameter_vectors()
    compute(shred("letter.txt"), e, s)





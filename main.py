from preProcessing import *
from models import *

test_file = open("test.txt").read().splitlines()
train_file = open("train.txt").read().splitlines()

# pre-processing: adding 
preprocessed_test_file1 = add_padding_lowercase(test_file)
preprocessed_train_file1 = add_padding_lowercase(train_file)

# dictionary with ALL tokens in training data
training_word_dictionary = count_words_in_train_file(preprocessed_train_file1)

# replace training words that occur once
preprocessed_train_file = replace_one_word_occurrance(preprocessed_train_file1, training_word_dictionary)
preprocessed_test_file = replace_words_not_in_training(preprocessed_test_file1, preprocessed_train_file)

##################################################################
# testing models
unigram_dictionary = unigram(training_word_dictionary)

# change this back to train file NOT test file
bigram_dictionary = bigram(count_words_in_train_file(preprocessed_test_file), preprocessed_test_file, False)

######################################################################################################
q1 = count_words_in_train_file(preprocessed_train_file)
q1_= len(q1)
print (f"question 1: {q1_} \n")

q2 = sum(q1.values())-q1["<s>"]
print(f"question 2: {q2} \n")

print("\n q3: ")
dict_test = unigram(count_words_in_train_file(preprocessed_test_file1))
dict_train = unigram(count_words_in_train_file(preprocessed_train_file1))

q3 = find_percentage_tokens(dict_test,dict_train)
print(f"question 3| tokens not occurred:  {q3[0]}") 
print(f"question 3| types not occurred: {q3[1]}")

print("\n q4: ")
test_file_without_s = preprocessed_test_file.replace('<s>', "")
train_file_without_s = preprocessed_train_file.replace('<s>', "")
dict_test = bigram(count_words_in_train_file(test_file_without_s), test_file_without_s, False)
dict_train = bigram(count_words_in_train_file(train_file_without_s), train_file_without_s, False)
q4 = find_percentage_tokens(dict_test,dict_train)
print(f"question 4| tokens not occurred:  {q4[0]}")
print(f"question 4| types not occurred: {q4[1]}")

print("\n q5 and q6: ")
# unigram
processed_unigram = unigram(count_words_in_train_file(preprocessed_train_file))

# unique_words = unigram(training_word_dictionary)
sentence = "I look forward to hearing you reply . "
q5_and_q6_unigram = unigram_log_perplexity(sentence, processed_unigram)
# perplexity, total log
print(f"question 5| unigram total log probability:  {q5_and_q6_unigram[1]}")
print(f"question 6| unigram perplexity: {q5_and_q6_unigram[0]} \n")

bigram_dict = count_words_in_train_file(train_file_without_s)
bigram_MLE_without_smoothing = bigram(bigram_dict, train_file_without_s, False)
bigram_MLE_with_smoothing = bigram(bigram_dict, train_file_without_s, True)

# bigram without smoothing
q5_and_q6_bigram_without_smoothing= bigram_log_perplexity(sentence, bigram_MLE_with_smoothing, processed_unigram, False)
print(f"Question 5| total log prob. bi-gram without smoothing: {q5_and_q6_bigram_without_smoothing[0]}")
print(f"parameters with zero probability: {q5_and_q6_bigram_without_smoothing[1]}")
print(f"Question 6| perplexity for bigram without smoothing: {q5_and_q6_bigram_without_smoothing[2]} \n")

# bi gram with smoothing
q5_and_q6_bigram_with_smoothing= bigram_log_perplexity(sentence, bigram_MLE_with_smoothing, processed_unigram, True)
print(f"Question 5| total log prob. bi-gram with smoothing: {q5_and_q6_bigram_with_smoothing[0]}")
print(f"parameters with zero probability: {q5_and_q6_bigram_with_smoothing[1]}")
print(f"Question 6| perplexity for bigram withsmoothing: {q5_and_q6_bigram_with_smoothing[2]} \n")

print("\n q7: ")
# unigram
processed_unigram = unigram(count_words_in_train_file(preprocessed_train_file))
q7_unigram_perplexity = perplexity_unigram(preprocessed_test_file, processed_unigram)
print(f"Question 7| unigram perplexity: {q7_unigram_perplexity}")

#bigram without smoohting
preprocessed_test_file = preprocessed_test_file.replace('<s>', "")
q7_perplexity_bigram_without_smoothing = perplexity_bigram(preprocessed_test_file, bigram_MLE_without_smoothing, processed_unigram, False)
print(f"Question 7| bi gram without smoothing perplexity: {q7_perplexity_bigram_without_smoothing}")

#bigram with smoohting
preprocessed_test_file = preprocessed_test_file.replace('<s>', "")
q7_perplexity_bigram_with_smoothing = perplexity_bigram(preprocessed_test_file, bigram_MLE_with_smoothing, processed_unigram, True)
print(f"Question 7| bi gram with smoothing perplexity: {q7_perplexity_bigram_with_smoothing}")

print("finished")
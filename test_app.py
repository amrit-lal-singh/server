from flask import Flask, request, jsonify

import json
import numpy as np

# Predefined phrase list
phrase_list = ["don't know how to start","how to start", "how start", "moving objects", "FBD", "nlm", "laws of motion", "red block", "yellow block", "blue block", "green block"
               , "purple block", "pink string", "red string", "white string", "yellow string", "How many moving objects", "many strings", "imaginary sphere",
               "each moving block", "strings", "string","force at this point of intersection","intersection","y-direction","FBD equation","how many points","cuts the strings"
               ,"second pulley","rate of change","length of string","left","right","human","only","and","Net","number of parts which change length","change length","number of parts",
               "double derivative","l₁","l₂","first","second","third","fourth","first pulley","moving pulley","l₃","constraint equation","What are the final accelerations, tensions and normal reactions","final",
               "contacting surfaces", "surfaces", "contacting", "normal", "reaction", "block 1", "incline","perpendicular",  "x-direction", "first", "second", "third","final", "constraint", "big"]

def check_phrases(sentence, phrase_list):
    presence_list = []
    formatted_sentence = sentence.lower().replace(" ", "")
    
    for phrase in phrase_list:
        formatted_phrase = phrase.lower().replace(" ", "")
        if formatted_phrase in formatted_sentence:
            presence_list.append(1)
        else:
            presence_list.append(0)
    
    return presence_list

# # Load JSON data from a file
# with open('/home/amrit/physics/server/data/68.json', 'r') as json_file:
#     data = json.load(json_file)

# Process the JSON data and generate binary lists
# binary_lists = []

# for step in data:
#     for question in step['stepQuestions']:
#         question_text = question['question']
#         binary_list = check_phrases(question_text, phrase_list)
#         binary_lists.append(binary_list)

# # Display questions and the generated binary lists
# for i, (question_text, binary_list) in enumerate(zip((q['question'] for step in data for q in step['stepQuestions']), binary_lists), start=1):
#     print(f"Question {i}: {question_text}")
#     print(f"Binary List {i}: {binary_list}")
#     print("=" * 50)  # Separating line for clarity


# # Provided sentence
# sentence = "What is double derivative of change in l₁ for red string due to yellowblock"
# # Process the sentence and generate binary list
# binary_output = check_phrases(sentence, phrase_list)

# Print the sentence and its binary output
# print(f"Sentence: {sentence}")
# print(f"Binary Output: {binary_output}")


##################################################


def hamming_distance(bin_list1, bin_list2):
    return np.sum(np.abs(np.array(bin_list1) - np.array(bin_list2)))

import json
import numpy as np

# ... (your code remains the same up to this point) ...

def find_most_similar_question(binary_output, binary_lists, data):
    # Initialize variables to keep track of best match
    best_match_index = -1
    best_hamming_distance = float('inf')  # Initialize with a large value

    # Compare the binary output of the sentence with binary outputs of questions
    for i, question_binary in enumerate(binary_lists):
        distance = hamming_distance(binary_output, question_binary)
        if distance < best_hamming_distance:
            best_hamming_distance = distance
            best_match_index = i

    # Calculate the total number of questions in the JSON data
    total_questions = sum(len(step['stepQuestions']) for step in data)

    # Determine the stepNumber and index of the best match question
    step_number = -1
    question_index = -1

    if best_match_index != -1 and best_match_index < total_questions:
        step_index = 0
        while best_match_index >= len(data[step_index]['stepQuestions']):
            best_match_index -= len(data[step_index]['stepQuestions'])
            step_index += 1

        step_number = step_index
        question_index = best_match_index

    return step_number, question_index


def find_threejsstep(most_similar_question, data):
    for step in data:
        for question in step['stepQuestions']:
            if question['question'] == most_similar_question:
                return question['threejsstep']
    return None

def find_step_question_association(sentence, phrase_list, binary_lists, data):
    # Process the sentence and generate binary output
    binary_output = check_phrases(sentence, phrase_list)

    # Find the stepNumber and question index
    step_number, question_index = find_most_similar_question(binary_output, binary_lists, data)

    # Determine0
    # ?>the most similar question text
    most_similar_question = None
    if step_number != -1 and question_index != -1:
        most_similar_question = data[step_number]['stepQuestions'][question_index]['question']

    # Find associated threejsstep
    associated_threejsstep = find_threejsstep(most_similar_question, data)

    return step_number, question_index, associated_threejsstep

def find_and_print_associated_info(sentence, phrase_list, binary_lists, data):
    # Call the function to find step number, question index, and associated threejsstep
    step_number, question_index, associated_threejsstep = find_step_question_association(sentence, phrase_list, binary_lists, data)

    if step_number != -1:
        print(f"Question found in Step {step_number}, Index {question_index} within that step.")
        if associated_threejsstep is not None:
            print("Associated threejsstep:", associated_threejsstep)
        else:
            print("No associated threejsstep found for the given question.")
    else:
        print("No similar question found")
    return step_number, question_index
# sentence = "What is the force at this point of intersection on second pulley?"
# # Call the function to find and print associated information
# step , index = find_and_print_associated_info(sentence, phrase_list, binary_lists, data)
# # Provided sentence

# if binary_list[0] == 0 or binary_list[1] == 0 or binary_list[2] ==0:
    # return 0, 0
    # step = 0
    # index = 0


app = Flask(__name__)

@app.route('/predict_step', methods=['POST'])
def find_step_question():
    data2 = request.get_json()
    sentence = data2['text']
    question_num = data2.get('questionNumber')
    
    # Load JSON data from a file
    path = "/home/amrit/physics/server/data/"+str(question_num)+".json"
    with open(path, 'r') as json_file:
        data = json.load(json_file)
        
        
    binary_lists = []
    binary_output = check_phrases(sentence, phrase_list)
    for step in data:
        for question in step['stepQuestions']:
            question_text = question['question']
            binary_list = check_phrases(question_text, phrase_list)
            binary_lists.append(binary_list)
    # Call the function to find and print associated information
    step, index = find_and_print_associated_info(sentence, phrase_list, binary_lists, data)
    print(binary_output)
    
    
    
    if question_num == "68":
        if binary_output[0] == 1 or binary_output[1] == 1 or binary_output[2] ==1:
        # return 0, 0
            step = 0
            index = 0
        # Create the response JSON
        if binary_output[4] == 1 and binary_output[7] == 1:
        # return 0, 0
            step = 2
            index = 0
        # Create the response JSON
        if binary_output[-1] == 1 and binary_output[4] == 1:
        # return 0, 0
            step = 2
            index = 14
        if binary_output[12] == 1 and binary_output[-2] == 1:
        # return 0, 0
            step = 3
            index = 0  
        if binary_output[-3] == 1:
        # return 0, 0
            step = 4
            index = 0 
            
            
    if question_num == "11":
            
        if binary_output[0] == 1 or binary_output[1] == 1 or binary_output[2] ==1:
        # return 0, 0
            step = 0
            index = 0
        # Create the response JSON
        if binary_output[4] == 1 and binary_output[7] == 1:#red fbd
        # return 0, 0
            step = 1
            index = 0
        # Create the response JSON
        if binary_output[8] == 1 and binary_output[4] == 1:#yellow fbd
        # return 0, 0
            step = 1
            index = 3
        if binary_output[12] == 1 and binary_output[-2] == 1:#pink constraint
        # return 0, 0
            step = 2 #constraint step index is 2 cause understanding is removed
            index = 0  
        if binary_output[-3] == 1:
        # return 0, 0
            step = 3
            index = 0 
            
    if question_num == "12":
            
        if binary_output[0] == 1 or binary_output[1] == 1 or binary_output[2] ==1:
        # return 0, 0
            step = 0
            index = 0
        # Create the response JSON
        if binary_output[4] == 1 and binary_output[7] == 1:#red fbd
        # return 0, 0
            step = 1
            index = 0
        # Create the response JSON
        if binary_output[8] == 1 and binary_output[4] == 1:#yellow fbd
        # return 0, 0
            step = 1
            index = 3
        if binary_output[12] == 1 and binary_output[-2] == 1:#pink constraint
        # return 0, 0
            step = 2 #constraint step index is 2 cause understanding is removed
            index = 2  
        if binary_output[-3] == 1:
        # return 0, 0
            step = 3
            index = 0 
    if question_num == "13":
            
        if binary_output[0] == 1 or binary_output[1] == 1 or binary_output[2] ==1:
        # return 0, 0
            step = 0
            index = 0
        # Create the response JSON
        if binary_output[4] == 1 and binary_output[7] == 1:#red fbd
        # return 0, 0
            step = 1
            index = 0
        # Create the response JSON
        if binary_output[8] == 1 and binary_output[4] == 1:#yellow fbd
        # return 0, 0
            step = 1
            index = 3
        if binary_output[12] == 1 and binary_output[-2] == 1:#pink constraint
        # return 0, 0
            step = 2 #constraint step index is 2 cause understanding is removed
            index = 0
        if binary_output[-3] == 1:
        # return 0, 0
            step = 3
            index = 0 
    if question_num == "14":
            
        if binary_output[0] == 1 or binary_output[1] == 1 or binary_output[2] ==1:
        # return 0, 0
            step = 0
            index = 0
        # Create the response JSON
        if binary_output[4] == 1 and binary_output[7] == 1:#red fbd
        # return 0, 0
            step = 1
            index = 0
        # Create the response JSON
        if binary_output[8] == 1 and binary_output[4] == 1:#yellow fbd
        # return 0, 0
            step = 1
            index = 3
        if binary_output[12] == 1 and binary_output[-2] == 1:#pink constraint
        # return 0, 0
            step = 2 #constraint step index is 2 cause understanding is removed
            index = 0
        if binary_output[-3] == 1:
        # return 0, 0
            step = 3
            index = 0 
    if question_num == "40":
            
        if binary_output[0] == 1 or binary_output[1] == 1 or binary_output[2] ==1:
        # return 0, 0
            step = 0
            index = 0
        # Create the response JSON
        if binary_output[4] == 1 and binary_output[7] == 1:#red fbd
        # return 0, 0
            step = 1
            index = 0
        # Create the response JSON
        if binary_output[8] == 1 and binary_output[4] == 1:#yellow fbd
        # return 0, 0
            step = 1
            index = 6
        if binary_output[12] == 1 and binary_output[-2] == 1:#pink constraint
        # return 0, 0
            step = 2 #constraint step index is 2 cause understanding is removed
            index = 0
        if binary_output[-3] == 1:
        # return 0, 0
            step = 3
            index = 0 

    if question_num == "41":
            
        if binary_output[0] == 1 or binary_output[1] == 1 or binary_output[2] ==1:
        # return 0, 0
            step = 0
            index = 0
        # Create the response JSON
        if binary_output[4] == 1 and binary_output[7] == 1:#red fbd
        # return 0, 0
            step = 1
            index = 0
        # Create the response JSON
        if binary_output[8] == 1 and binary_output[4] == 1:#yellow fbd
        # return 0, 0
            step = 1
            index = 6
        if binary_output[12] == 1 and binary_output[-2] == 1:#pink constraint
        # return 0, 0
            step = 2 #constraint step index is 2 cause understanding is removed
            index = 0
        if binary_output[-3] == 1:
        # return 0, 0
            step = 3
            index = 0 

    if question_num == "42":
            
        if binary_output[0] == 1 or binary_output[1] == 1 or binary_output[2] ==1:
        # return 0, 0
            step = 0
            index = 0
        # Create the response JSON
        if binary_output[4] == 1 and binary_output[7] == 1:#red fbd
        # return 0, 0
            step = 1
            index = 0
        # Create the response JSON
        if binary_output[8] == 1 and binary_output[4] == 1:#yellow fbd
        # return 0, 0
            step = 1
            index = 6
        if binary_output[12] == 1 and binary_output[-2] == 1:#pink constraint
        # return 0, 0
            step = 2 #constraint step index is 2 cause understanding is removed
            index = 0
        if binary_output[-3] == 1:
        # return 0, 0
            step = 3
            index = 0 
            

    if question_num == "43":
            
        if binary_output[0] == 1 or binary_output[1] == 1 or binary_output[2] ==1:
        # return 0, 0
            step = 0
            index = 0
        # Create the response JSON
        if binary_output[4] == 1 and binary_output[7] == 1:#red fbd
        # return 0, 0
            step = 1
            index = 0
        # Create the response JSON
        if binary_output[8] == 1 and binary_output[4] == 1:#yellow fbd
        # return 0, 0
            step = 1
            index = 6
        if binary_output[12] == 1 and binary_output[-2] == 1:#pink constraint
        # return 0, 0
            step = 2 #constraint step index is 2 cause understanding is removed
            index = 0
        if binary_output[-3] == 1:
        # return 0, 0
            step = 3
            index = 0 
    if question_num == "44":
            
        if binary_output[0] == 1 or binary_output[1] == 1 or binary_output[2] ==1:
        # return 0, 0
            step = 0
            index = 0
        # Create the response JSON
        if binary_output[4] == 1 and binary_output[7] == 1:#red fbd
        # return 0, 0
            step = 1
            index = 0
        # Create the response JSON
        if binary_output[8] == 1 and binary_output[4] == 1:#yellow fbd
        # return 0, 0
            step = 1
            index = 6
        if binary_output[12] == 1 and binary_output[-2] == 1:#pink constraint
        # return 0, 0
            step = 2 #constraint step index is 2 cause understanding is removed
            index = 0
        if binary_output[-3] == 1:
        # return 0, 0
            step = 3
            index = 0             
    response = {
        'predicted_step': step,
        'predicted_substep': index
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

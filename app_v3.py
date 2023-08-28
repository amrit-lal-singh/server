from flask import Flask, request, jsonify
app = Flask(__name__)

# Predefined phrase list
predefined_phrase_list = ["how to start","how start","how many objects", "FBD","nlm","laws of motion","red block", "yellow block", "blue block", "green block", "purple block", "pink string", "red string", "white string", "yellow string","How many moving objects", "How many strings" , "imaginary sphere", ]

def check_phrases(sentence, phrase_list):
    presence_list = []
    formatted_sentence = sentence.lower().replace(" ", "")
    
    for phrase in phrase_list:\
        
        
        formatted_phrase = phrase.lower().replace(" ", "")
        if formatted_phrase in formatted_sentence:
            presence_list.append(1)
        else:
            presence_list.append(0)
    
    return presence_list

@app.route('/check', methods=['POST'])
def check():
    data = request.json
    sentence = data.get('sentence', '')
    
    if not sentence:
        return jsonify({'error': 'Invalid input'}), 400
    
    result = check_phrases(sentence, predefined_phrase_list)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)

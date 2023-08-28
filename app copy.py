from flask import Flask, request, jsonify
import numpy as np 
import pandas as pd
# import tokenization
import tensorflow as tf
# from tensorflow.python.keras.preprocessing.text import tokenization
import tensorflow_hub as hub
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


# m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer("/home/amrit/physics/server/bert_en_uncased_L-12_H-768_A-12_4", trainable=True)

from flask import Flask, request, jsonify
import numpy as np 
import pandas as pd
# import tokenization
import tensorflow as tf
# from tensorflow.python.keras.preprocessing.text import tokenization
import tensorflow_hub as hub
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


# m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer("/home/amrit/physics/server/bert_en_uncased_L-12_H-768_A-12_4", trainable=True)





# Assuming `bert_layer` is the BERT model layer from the transformers library
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocab_file, do_lower_case)

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len-len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)





def predict_label(text, model, tokenizer, max_len=250):
    # Tokenize the input text
    input_data = bert_encode([text], tokenizer, max_len=max_len)
    
    # Make predictions using the model
    predictions = model.predict(input_data)
    
    # Convert predictions to class labels
    label_index = np.argmax(predictions[0])
#     label = label.classes_[label_index]

    return label_index


def predict_label2(text, model2, tokenizer, max_len=250):
    # Tokenize the input text
    input_data = bert_encode([text], tokenizer, max_len=max_len)
    
    # Make predictions using the model
    predictions = model2.predict(input_data)
    
    # Convert predictions to class labels
    label_index = np.argmax(predictions[0])
#     label = label.classes_[label_index]
    
    return label_index











loaded_model = tf.keras.models.load_model('/home/amrit/physics/server/saved_model')
loaded_model2 = tf.keras.models.load_model('/home/amrit/physics/server/saved_model2')

# Sample input text for prediction
input_text = "What are the final accelerations, tensions, and normal reactions?"

predicted_label = predict_label(input_text, loaded_model, 
                                tokenizer, max_len=250)
print(f"Predicted Label: {predicted_label}")


app = Flask(__name__)

# Load the pre-trained model and tokenizer
# model_name = "bert-base-uncased"  # Replace with your model name
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/predict_step', methods=['POST'])
def predict_step():
    try:
        data = request.get_json()
        input_text = data.get('text')
        # Sample input text for prediction
        # input_text = "What are the final accelerations, tensions, and normal reactions?"
        # Predict the step and substep
        predicted_step = predict_label(input_text, loaded_model, tokenizer)
        predicted_substep = predict_label2(input_text, loaded_model2, tokenizer)

        # Convert predictions to integers and create a JSON response
        response_data = {
            'predicted_step': int(predicted_step) + 1,  # Adding 1 to match your requirement
            'predicted_substep': int(predicted_substep) + 1
        }
        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)





# Assuming `bert_layer` is the BERT model layer from the transformers library
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocab_file, do_lower_case)

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len-len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)





def predict_label(text, model, tokenizer, max_len=250):
    # Tokenize the input text
    input_data = bert_encode([text], tokenizer, max_len=max_len)
    
    # Make predictions using the model
    predictions = model.predict(input_data)
    
    # Convert predictions to class labels
    label_index = np.argmax(predictions[0])
#     label = label.classes_[label_index]

    return label_index


def predict_label2(text, model2, tokenizer, max_len=250):
    # Tokenize the input text
    input_data = bert_encode([text], tokenizer, max_len=max_len)
    
    # Make predictions using the model
    predictions = model2.predict(input_data)
    
    # Convert predictions to class labels
    label_index = np.argmax(predictions[0])
#     label = label.classes_[label_index]
    
    return label_index











loaded_model = tf.keras.models.load_model('/home/amrit/physics/server/saved_model')
loaded_model2 = tf.keras.models.load_model('/home/amrit/physics/server/saved_model2')

# Sample input text for prediction
input_text = "What are the final accelerations, tensions, and normal reactions?"

predicted_label = predict_label(input_text, loaded_model, 
                                tokenizer, max_len=250)
print(f"Predicted Label: {predicted_label}")


app = Flask(__name__)

# Load the pre-trained model and tokenizer
# model_name = "bert-base-uncased"  # Replace with your model name
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/predict_step', methods=['POST'])
def predict_step():
    try:
        data = request.get_json()
        input_text = data.get('text')
        # Sample input text for prediction
        # input_text = "What are the final accelerations, tensions, and normal reactions?"
        # Predict the step and substep
        predicted_step = predict_label(input_text, loaded_model, tokenizer)
        predicted_substep = predict_label2(input_text, loaded_model2, tokenizer)

        # Convert predictions to integers and create a JSON response
        response_data = {
            'predicted_step': int(predicted_step) + 1,  # Adding 1 to match your requirement
            'predicted_substep': int(predicted_substep) 
        }
        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

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


m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(m_url, trainable=True)






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






import numpy as np

def predict_label(text, model, tokenizer, max_len=250):
    # Tokenize the input text
    input_data = bert_encode([text], tokenizer, max_len=max_len)
    
    # Make predictions using the model
    predictions = model.predict(input_data)
    
    # Convert predictions to class labels
    label_index = np.argmax(predictions[0])
#     label = label.classes_[label_index]
    
    return label_index
loaded_model = tf.keras.models.load_model('/home/amrit/physics/server/saved_model')

# Sample input text for prediction
input_text = "What are the final accelerations, tensions, and normal reactions?"

predicted_label = predict_label(input_text, loaded_model, 
                                tokenizer, max_len=250)
print(f"Predicted Label: {predicted_label}")
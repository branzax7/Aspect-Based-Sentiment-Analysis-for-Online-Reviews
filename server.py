from flask import Flask, request, jsonify, Response, render_template_string, render_template
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BartTokenizer, BartModel
import os
import ast
from openai import OpenAI
import re

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="xxx",
)
     

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('my_model_BART.keras')


with open("model_raw_data_2", "rb") as fp:   # Unpickling
    bart_data = pickle.load(fp)
_, X_bart,  _, _, _, _, _, _ = bart_data

max_bart = [e for e in X_bart if e.shape[1] == 72]

with open("model_proccess_data_3", "rb") as fp:   # Unpickling
    model_data = pickle.load(fp)
X_train_bart, X_test_bart, y1_train_hot, y1_test_hot, y2_train_hot, y2_test_hot, encoder1, encoder2 = model_data

with open("test_output_data_3", "rb") as fp:   # Unpickling
    test_data = pickle.load(fp)
X_train, X_test, y1_train, y1_test, y2_train, y2_test = test_data



def text_to_sentences(text):
    return [sentence for sentence in re.split(r'[.!?]', text) if sentence != '']

def del_unwanted_data(data):
    for index in range(len(data)):
        if len(data[index]) > 1:
            data[index] = data[index][1]
    return data

def proccess_gpt_result(result):
    message_content = result.choices[0].message.content
    message_content = message_content.replace('\n', '')
    start_tuple = message_content.find('(')
    end_tuple = message_content.find(')')
    next_t = message_content[start_tuple+1:].find('(')
    if next_t != -1 and next_t + start_tuple + 1 < end_tuple:
        end_tuple = message_content[end_tuple+1:].find(')') + end_tuple + 1
        
    data = []
    while start_tuple != -1 and end_tuple != -1 and start_tuple < end_tuple:
        data.append(ast.literal_eval(message_content[start_tuple: end_tuple+1]))
        # print(data)
        message_content = message_content[end_tuple+1:]
        start_tuple = message_content.find('(')
        end_tuple = message_content.find(')')
        next_t = message_content[start_tuple+1:].find('(')
        if next_t != -1 and next_t + start_tuple + 1 < end_tuple:
            # print("->",message_content, next_t + start_tuple, message_content[next_t + start_tuple + 1])
            # print("->",end_tuple, message_content[end_tuple], message_content[:end_tuple+1])
            end_tuple = message_content[end_tuple+1:].find(')') + end_tuple + 1
            # print("->",end_tuple, message_content[end_tuple], message_content[:end_tuple+1])
    return data


def get_clauses_from_gpt(review_text):
    # Construct the input text with the review and opinions
    # opinions_text = "\n".join([f'Category: {op["category"]}, Polarity: {op["polarity"]}' for op in opinions])
    # opinions_text = "\n".join([f'Target: {op["target"]}, Category: {op["category"]}, Polarity: {op["polarity"]}' for op in opinions])
    prompt = f"Split the following sentence into clauses, each caluses must contain at least a noun, \
    a verb and an adjective or adverb if possible. Differnt clauses can have the same verb, adjective or adverb, \
    but cant have the same noun. Some sentences cant be splited into caluses, so it will be only one caluse which \
    will be the whole sentence. Keep the output format! \n\nSentence: {review_text}\n\n\
    Output format:[(1, \"<clause_text>\")]"

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that helps to process text by splitting\
              sentences into clauses"},
            {"role": "user", "content": prompt}
        ]
    )

    # return response['choices'][0]['message']['content'].strip()
    return response


def vectorize(clause):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartModel.from_pretrained('facebook/bart-large')

    inputs = tokenizer(clause, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state

    return last_hidden_state

def proccess_bart_data(clause_vectors_res):
    tf.squeeze(clause_vectors_res[0], axis=0)
    clause_vectors_res_2d = [tensor.squeeze(0) for tensor in clause_vectors_res]
    clause_vectors_padded = pad_sequence(clause_vectors_res_2d, batch_first=True)
    return clause_vectors_padded


def check_model(review):
    sentences = text_to_sentences(review)
    clauses = []
    for sentence in sentences:
        clauses += del_unwanted_data(proccess_gpt_result(get_clauses_from_gpt(sentence)))
    # result = get_clauses_from_gpt("The wine list is interesting and has many good values.")
    # proccess_gpt_result(result)
    print("sentences=",sentences)
    print("clauses=", clauses)


    results = []

    for clause in clauses:
        clause_bart = vectorize(clause)
        vec = [clause_bart] + max_bart
        vec = proccess_bart_data(vec)
        single_input = np.expand_dims(vec[0], axis=0)

        # Get predictions
        predictions = model.predict(single_input)

        # Assuming the model has two outputs
        y1_pred = predictions[0]
        y2_pred = predictions[1]

        # Decode the one-hot predictions back to original labels
        predicted_category = encoder1.inverse_transform(y1_pred)
        predicted_sentiment = encoder2.inverse_transform(y2_pred)

        result = {
            "Input": clause,
            "Predicted Category": predicted_category[0][0],
            "Predicted Sentiment": predicted_sentiment[0][0]
        }
        results.append(result)
    
    return results

def check_with_test_data(number):
    single_input = np.expand_dims(X_test_bart[number], axis=0)

    # Get predictions
    predictions = model.predict(single_input)

    # Assuming the model has two outputs
    y1_pred = predictions[0]
    y2_pred = predictions[1]

    # Decode the one-hot predictions back to original labels
    predicted_category = encoder1.inverse_transform(y1_pred)
    predicted_sentiment = encoder2.inverse_transform(y2_pred)

    result = {
        "Input": X_test[number],
        "Predicted Category": predicted_category[0][0],
        "Predicted Sentiment": predicted_sentiment[0][0]
    }
    
    return result

@app.route("/", methods=["GET", "POST"])
def demo():
    text = ""
    if request.method == "POST":
        if 'text' in request.form:
            text = request.form['text']
            result = check_model(text)
            return render_template_string(template, result=result, text=text, max_value=len(X_test)-1)
        # elif 'number' in request.form:
        #     number = request.form['number']
        #     result = check_with_test_data(int(number))
        #     return render_template_string(template, result=result, text=text, max_value=len(X_test)-1)
    return render_template_string(template, text=text, max_value=len(X_test)-1)

template = """
<!doctype html>
<html>
<head><title>Aspect-Based Sentiment Analysis for Online Reviews</title></head>
<body>
    <h1>Aspect-Based Sentiment Analysis for Online Reviews</h1>
    <form method="post">
        <label for="text">Enter text:</label>
        <textarea id="text" name="text" rows="4" cols="50">{{ text }}</textarea>
        <input type="submit" value="Submit">
    </form>
    {% if result %}
    <h2>Results</h2>
    {% for res in result %}
        <p>Clause: {{ res['Input'] }}</p>
        <p>Predicted Category: {{ res['Predicted Category'] }}</p>
        <p>Predicted Sentiment: {{ res['Predicted Sentiment'] }}</p>
        <hr>
    {% endfor %}
    {% endif %}
</body>
</html>
"""

if __name__ == '__main__':
    app.run('0.0.0.0', 7020, debug=True)

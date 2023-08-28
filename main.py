from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import difflib

NUM_OF_CLASSES = 11
MAX_SIZE_OF_LINE = 10
VOCAB_SIZE = 1000
OUTPUT_DIM = 10
CODING = {
    1: 'Author',
    2: 'Title',
    3: 'Journal',
    4: 'Year',
    5: 'Volume',
    6: 'Release',
    7: 'Start page',
    8: 'End page',
    9: 'Link',
    0: 'etc'
}


def similar(seq1, seq2):
    return difflib.SequenceMatcher(a=seq1.lower(), b=seq2.lower()).ratio()


def from_sentence_to_nums(string):
    list_made_of_attributes = list()
    sent = dict()

    for index, word in enumerate(string.split()):
        attributes = dict()
        if len(word) == 1 and word in ['-', '//']:
            attributes['Symbol'] = 0
            attributes['Uppercase'] = 0
            if word == '//':
                attributes['Separators'] = 2
            elif word == '-':
                attributes['Separators'] = 1
            else:
                attributes['Separators'] = 0
            attributes['DotOrComma'] = 0
            attributes['Initial'] = 0

        elif word.isalpha():
            attributes['Symbol'] = 1
            if word[0].isupper():
                attributes['Uppercase'] = 2
            elif word[0].islower():
                attributes['Uppercase'] = 1
            else:
                attributes['Uppercase'] = 0
            attributes['Separators'] = 0
            attributes['DotOrComma'] = 0
            attributes['Initial'] = 0

        elif word.isdigit():
            attributes['Symbol'] = 2
            attributes['Uppercase'] = 0
            attributes['Separators'] = 0
            attributes['DotOrComma'] = 0
            attributes['Initial'] = 0

        else:
            attributes['Symbol'] = 3
            attributes['Uppercase'] = 0
            if '//' in word:
                attributes['Separators'] = 2
            elif '-' in word:
                attributes['Separators'] = 1
            else:
                attributes['Separators'] = 0

            if '.' in word:
                attributes['DotOrComma'] = 1
            elif ',' in word:
                attributes['DotOrComma'] = 2
            else:
                attributes['DotOrComma'] = 0

            attributes['Initial'] = 1

        list_made_of_attributes.append(attributes)
    new_list_made_of_attributes = list()

    for i, attributes in enumerate(list_made_of_attributes):
        new_attributes = dict()
        if i == 0:
            for att in attributes.keys():
                new_attributes['Prev' + att] = 0
        else:
            for att in attributes.keys():
                new_attributes['Prev' + att] = list_made_of_attributes[i - 1][att]

        for att in attributes.keys():
            new_attributes[att] = list_made_of_attributes[i][att]

        if i == (len(list_made_of_attributes) - 1):
            for att in attributes.keys():
                new_attributes['Next' + att] = 0
        else:
            for att in attributes.keys():
                new_attributes['Next' + att] = list_made_of_attributes[i + 1][att]

        new_list_made_of_attributes.append(list(new_attributes.values()))
    return new_list_made_of_attributes


def concatenation(data):
    heads = data.keys()
    new_data = data[heads[0]]

    for head in heads[1:]:
        new_data = new_data + '\t' + data[head].apply(str)

    return new_data


def biblio_matrix(source, classes):
    lst = list()
    source_one_string = source[:].replace(' ', '')
    source_splitted = source[:].split()
    classes = [cl.split() for k, cl in enumerate(classes)]

    matrix = [[0 for j in range(len(source_splitted))] for i in range(len(classes))]

    for k in range(len(classes)):
        try:
            num_words = len(classes[k])
            probabilities = [similar(classes[k][0], string) for string in source_splitted]
            index = probabilities.index(max(probabilities))

            for i in range(num_words):
                matrix[k][index + i] = k + 1
        except Exception as e:
            continue

    for i in range(len(matrix[0])):
        matrix[0][i] = max([j[i] for j in matrix])

    return tf.cast(matrix[0], tf.float32)


def map_record_to_training_data(record):
    record = record.split('\t')
    source = record[0]
    classes = record[1:]
    print(classes)
    return source, classes


def one_column(dataframe):
    return dataframe['Источник']


def predict_nn(data, model):
    output = model.predict(tf.cast(data, tf.float32))
    prediction = np.argmax(output, axis=-1)
    return prediction


def start_nn(data, model):
    data_list = list()
    for line in data:
        dictionary_of_data = {'String': line,
                              'After_nn': predict_nn(from_sentence_to_nums(line), model)}

        dictionary_of_types = dict()
        for i in range(len(CODING)):
            dictionary_of_types[CODING[i]] = list()

        for num in range(len(CODING)):
            prediction = dictionary_of_data['After_nn']
            for index, word in enumerate(dictionary_of_data['String'].split()):
                if prediction[index] == num:
                    dictionary_of_types[CODING[num]].append(word)
        dictionary_of_data['Distribution'] = dictionary_of_types
        data_list.append(dictionary_of_data)

    # for d in data_list:
    #     print(d['String'])
    #     for key in d['Distribution'].keys():
    #         print(key, d['Distribution'][key])

    return data_list


df = pd.read_excel('C:/Users/okudz/Desktop/mirea/1000_tests.xlsx')[826:]

sources = one_column(df).tolist()

model = keras.models.load_model('model_first4')

data = start_nn(sources, model)
print(data)

dict_to_df = {'String': [data[i]['String'] for i in range(len(data))]}

for i in range(len(CODING)):
    dict_to_df[CODING[i]] = list()

for num in range(len(data)):
    for i in range(len(CODING)):
        lst = data[num]['Distribution'][CODING[i]]
        new_string = ' '.join(lst) if lst else '-'
        print(new_string)
        dict_to_df[CODING[i]].append(new_string)


students = pd.DataFrame(dict_to_df)

df = pd.DataFrame(dict_to_df)

# displaying the DataFrame
print('DataFrame:\n', df)

# saving the DataFrame as a CSV file
gfg_csv_data = df.to_csv('russian_first4_columns.csv', index=True)

print('\nCSV String:\n', gfg_csv_data)

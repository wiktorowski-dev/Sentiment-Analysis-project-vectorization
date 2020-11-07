import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import json


class VectorCreator:
    def __init__(self):
        super(VectorCreator, self).__init__()

    def transform_text_to_vector(self, data_set_text, data_set_tags_array,
                                 test_size, max_length_output, num_words, random_state=77,
                                 glove_path=None,
                                 process_glove=True,
                                 embedding_matrix_file_name=None,
                                 embedding_key_to_val_file_name=None,
                                 save_glove_post_processing_files=False):

        # Validate input data
        if not process_glove:
            if not embedding_matrix_file_name or not embedding_key_to_val_file_name:
                # Missing file names
                raise Exception('{}.{} | missing embedding_matrix_file_name or embedding_key_to_val_file_name'.
                                format(VectorCreator.__name__,
                                       VectorCreator.transform_text_to_vector.__name__))

        else:
            if not glove_path:
                raise Exception('{}.{} | missing glove_path'.
                                format(VectorCreator.__name__,
                                       VectorCreator.transform_text_to_vector.__name__))

        # Separate data to training and negative
        train_text, test_text, train_tags, test_tags = \
            train_test_split(data_set_text, data_set_tags_array, test_size=test_size, random_state=random_state)

        # Create tokenizer to tokenize sentences
        tokenizer = Tokenizer(num_words=num_words)

        # Fit text into tokenizer
        tokenizer.fit_on_texts(train_text)

        # Change text to integer equivalent
        train_text = tokenizer.texts_to_sequences(train_text)
        test_text = tokenizer.texts_to_sequences(test_text)

        # Set length of each integer-sentence to 100
        train_text = pad_sequences(train_text, padding='post', maxlen=max_length_output)
        test_text = pad_sequences(test_text, padding='post', maxlen=max_length_output)

        # First element in matrix is 0

        if process_glove:
            output, embedding_dict = self.__create_glove_embedding(glove_path, save_glove_post_processing_files,
                                                                   embedding_matrix_file_name,
                                                                   embedding_key_to_val_file_name)
        else:
            output, embedding_dict = self.__load_glove_embedding(embedding_key_to_val_file_name,
                                                                 embedding_matrix_file_name)

        # Change tokens from tokenizer to tokens from embedding matrix
        train_text = self.from_tokenizer_code_to_glove_code(train_text, embedding_dict, tokenizer)
        test_text = self.from_tokenizer_code_to_glove_code(test_text, embedding_dict, tokenizer)

        vocab_size = len(output)

        return train_text, test_text, train_tags, test_tags, output, vocab_size

    @staticmethod
    def from_tokenizer_code_to_glove_code(input_text, embeddings_dict, tokenizer):
        post_train_text = []
        for sentence in input_text:
            post_word = []
            for word in sentence:
                if word == 0:
                    post_word.append(0)
                    continue
                word_name = tokenizer.index_word[word]
                word_post_glove_key = embeddings_dict.get(word_name.lower())
                if not word_post_glove_key:
                    post_word.append(0)
                    continue
                post_word.append(word_post_glove_key)
            post_train_text.append(post_word)

        post_train_text = np.asarray(post_train_text)
        return post_train_text

    @staticmethod
    def __create_glove_embedding(glove_path, save_glove_post_processing_files, embedding_matrix_file_name,
                                 embedding_key_to_val_file_name):
        output = [np.zeros(100)]
        embeddings_dict = dict()
        index = 0
        # Create embedding matrix from glove file
        with open(glove_path, encoding='UTF-8') as file:
            for line in file:
                record = line.split()
                word = record.pop(0)
                if len(record) != 100:
                    continue
                output.append(np.asarray(record, dtype='float32'))
                embeddings_dict[word] = index + 1
                index += 1
        output = np.asarray(output, dtype='float32')
        if save_glove_post_processing_files and embedding_matrix_file_name and embedding_key_to_val_file_name:
            with open(embedding_key_to_val_file_name, mode='a') as file:
                file.write(json.dumps(embeddings_dict))

            np.save(embedding_matrix_file_name, output)

        return output, embeddings_dict

    @staticmethod
    def __load_glove_embedding(embedding_key_to_val_file_name, embedding_matrix_file_name):
        # If we proceed this once, we can load this from the storage
        with open(embedding_key_to_val_file_name) as file:
            embedding_dict = json.load(file)

        output = np.load(embedding_matrix_file_name)
        return output, embedding_dict

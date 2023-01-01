""" adapted from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text
that has been run through Unidecode. For other data, you can modify
_characters.'''


def get_symbols(symbol_set):
    if symbol_set == 'ukrainian':
        _punctuation = '\'.,?! '
        _special = '-+'
        _letters = 'абвгґдежзийклмнопрстуфхцчшщьюяєії'
        symbols = list(_punctuation + _special + _letters)
    else:
        raise Exception("{} symbol set does not exist".format(symbol_set))

    return symbols

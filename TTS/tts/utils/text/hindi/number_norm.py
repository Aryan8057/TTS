# number, decimal, currency
""" from https://github.com/keithito/tacotron """

import re
from typing import Dict
from indic_numtowords import num2words 
import inflect


_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
# _currency_re = re.compile(r"(£|\$|¥)([0-9\,\.]*[0-9]+)")
# _ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"-?[0-9]+")


num_to_word_dict = {
    '0': 'ज़ीरो', '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार',
    '5': 'पाँच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ',
	'०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
}


def _remove_commas(m):
    return m.group(1).replace(",", "")

def _expand_number(m):
    num_str = m.group(0)  # Extract the matched number as a string
    num = int(num_str)
    if num < 0:
        return "नैगेटिव " + num2words(abs(num), lang='hi')
    else:
        return num2words(num, lang='hi')


# 1. Convert roman to devanagari numbers:
def convert_num_to_words(m):
    num = int(m)
    if num < 0:
        return "नैगेटिव " + num2words(abs(num), lang='hi')
    else:
        return num2words(num, lang='hi')


def expand_decimal_point(m):
	decimal_number = m.group(1)
	integer_part,fractional_part = decimal_number.split('.')
	spoken_integer_part = convert_num_to_words(integer_part)
	spoken_fractional_part = ' '.join(num_to_word_dict[digit] for digit in fractional_part)
	return spoken_integer_part + ' पॉइंट ' + spoken_fractional_part
    # return spoken_integer_part + ' point ' + spoken_fractional_part

def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text) 
    # text = re.sub(_currency_re, _expand_currency, text)
    text = re.sub(_decimal_number_re, expand_decimal_point, text)
    # text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    # print(text)
    return text

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import re
from fuzzywuzzy import fuzz

class Product:
    def __init__(self, type, param_names, borders):
        self.type = type
        self.borders = dict()
        self.params = dict() #[имя параметра: значение]
        for p_n, b in zip(param_names, borders):
            self.borders[p_n] = b
    def check_borders(self, name, value):
        if self.borders[name][0] <=value and self.borders[name][1] >=value:
            return True
        return False 
    def set_params(self, params):
            self.params = params

def apply_filters(img):
    gray_img = cv. cvtColor(img, cv.COLOR_BGR2GRAY)
    _, th_img = cv.threshold(gray_img, 200, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    res_img = cv.dilate(th_img,np.zeros((5,5),np.uint8),iterations = 1)
    plt.imshow(res_img)
    return res_img

def get_text_from_img(img):
    text = pytesseract.image_to_string(img, lang='rus')
    return text.lower()

def tokenize(text):
    return re.split('\s+|[-]', text) 

def find_variation(tokens, standard):
    for i in range(len(tokens)):
        if fuzz.ratio(standard, tokens[i]) > 65: 
            tokens[i] = standard
            return i
  
def find_float_from_begin(text):
    res = re.search(r'\d+[,.]?\d?', text)
    if res:
        return float(res.group().replace(',', '.'))
    return -1

def find_float_from_end(text):
    res = re.findall(r'\d+[,.]?\d?', text)
    if res:
        return float(res[-1].replace(',', '.'))
    return -1
        
def find_param_values(tokens, params, product):
    param_values = dict() # [(имя параметра, значение)]
    idxwidth = 6
    num = len(params)
    for i, param in zip(range(num), params):
        value = -1
        if param[1] == None:
            begin, end = -1, -1
            if i >= 0 and i < num-1 and params[i+1][1] != None:
                begin, end = params[i+1][1]-2*idxwidth, params[i+1][1]
            elif i == num-1 and params[i-1][1] != None:
                begin, end = params[i-1][1], params[i-1][1]+2*idxwidth
            if begin >=0 or end >=0:
                text_part = ' '.join(tokens[max(0, begin) : min(end, len(tokens))])
                value = find_float_from_end(text_part)
        else:
            text_part = ' '.join(tokens[param[1]: param[1]+idxwidth])
            value = find_float_from_begin(text_part)
        if value > 0:
            if not product.check_borders(param[0], value):
                if product.check_borders(param[0], 0.1 * value):
                    value = 0.1 * value
                else:
                    continue
            param_values[param[0]] = value
    product.set_params(param_values)

def get_type_from_text(tokens, type_names):
    type = None
    for w in type_names:
        if find_variation(tokens, w) != None:
            type = w
            break
    return type

def get_params_from_text(tokens, param_names, product):
    params = [] # [(имя параметра, индекс токена с именем в тексте/None)]
    for w in param_names:
        idx = find_variation(tokens, w)
        params.append((w, idx))
    find_param_values(tokens, params, product)

def get_data_from_img(img, type_names, param_names, products_borders):
    filters_img = apply_filters(img)
    text = get_text_from_img(filters_img)
    tokens = tokenize(text) 
    type = get_type_from_text(tokens, type_names)
    if type == None:
        #type = 'молоко'
        type = input('Что это? ' + ', '.join(type_names) + '? ')
        if type not in (type_names):
            print('Неизвестный продукт ')
            return None
    product = Product(type, param_names, products_borders[type])
    get_params_from_text(tokens, param_names, product)
    if len(product.params) < len(param_names):
        diff = set(param_names) - set([p.key() for p in product.params]) 
        print('Недостаточно данных: ' + ', '.join(diff))
        return None
    #plt.show()
    return product
import cv2 as cv
from data_extraction import get_data_from_img
from datetime import date, datetime
from machine_learning import get_answer

if __name__ == '__main__':
    type_names = ['сметана', 'сливки', 'закваска', 'молоко']
    param_names = ['жир', 'белок', 'углеводы']
    param_borders = [[[10,30], [2,3], [3,4]],
                        [[10,35], [1,3], [2,5]],
                        [[1,3], [3,4], [4,5]],
                        [[1,6], [2,4], [4,5]]]
    products_borders = dict()
    for p_n, p_b in zip(type_names, param_borders):
        products_borders[p_n] = p_b

    human_data = {'пол': 'Ж','возраст': 40,'вес': 55,'дополнительно': 'диета'}
    img = cv.imread('data/img_3.jpg')
    
    product = get_data_from_img(img, type_names, param_names, products_borders)
    if product != None:
        date_end = input('Введите дату окончания срока годности ДД.MM.ГГГГ ')
        delta = (datetime.strptime(date_end, '%d.%m.%Y').date() -  date.today()).days

        product_data = {'тип': product.type} | product.params | {'окончание срока годности': delta}

        answer = get_answer(human_data, product_data)
        print(answer)


        
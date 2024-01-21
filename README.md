# ML_LAB
 
### Описание программы 
По поданным на вход фотографии упаковки продукта, сроку годности, а также данным некоторого человека определить, в каком количестве рекомендуется употреблять этот продукт.

### Вход 
  + фотография в формате jpg состава продукта, на которой указано количество белков, жиров, углеводов;
  + дата окончания срока годности в формате ДД.ММ.ГГГГ;
  + описание человека с некоторыми параметрами.

### Выход 
Число, обозначающее, какое количество продукта в день рекомендуеся употреблять. Значение приведено в граммах.

### Требования к входным данным 
 #### Рассматриваемые типы продуктов
  + Молоко  
  + Сливки   
  + Сметана  
  + Другие кисломолочные продукты (кефир, ряженка и т.п.)  
 #### Фотография  
  + Разрешение фотографии не менее 1280x720 px  
  + На фотографии только один продукт  
  + Продукт сфотографирован с расстояния не более ~25 см  
  + Информация с составом продукта целиком находится в кадре  
  + Текст четко виден на фотографии, отсутствие размытости  
  + Равномерное освещение состава продукта без бликов/теней  
 #### Описание человека
  + Пол - M/Ж
  + Возраст - целое число от 1 до 90
  + Вес - целое число от 14 до 90
  + Дополнительная информация:
    - проблемы с давлением 
    - непереносимость лактозы 
    - диета
    - спортивная активность
    - проблемы с желудком
    - отсутствует  

### План решения задачи
1. Преобразование исходного изображения  
    - Морфологическая и пороговая фильтрация  
2. Выделение текста из изображения  
    - С использованием pytesseract  
3. Выделение числовых и категориальных характеристик из текста  
    - Поиск ключевых слов-типов продуктов и параметров с использованием расстояния Левенштейна  
    - Поиск чисел, следующих за ключевыми словами-параметрами с использованием регулярных выражений  
    - Дополнительный поиск чисел, если не все слова-параметры найдены  
    - Проверка и исправление некоторых численных данных в соответствии с диапазоном допустимых значений  
4. Предсказание результата на основе обученной модели  
    - Обучение модели с использованием текстовых данных в размере 4800 строк, 2/3  данных - обучающая выборка, 1/3 - тестовая
    - Обучение методом градиентного бустинга с использованием XGBoost

### Оценка результатов
  + Извлечение данных из фотографии:
    - Текст на фотографии расспозновалься плохо при следующих ситуациях: мелкий шрифт состава или шрифт с нестандартной стилистикой, блики на фотографии, пестрый дизайн упаковки, текст сфотографирован под наклоном.
    - Следствие: отсутствие части данных,строки меняются местами при извлечении, происходит склеивание слов, утрата точек в десятичных дробях.
    - Извлечение данных производилось плохо в случае возникновения проблем на предыдущем этапе, дополнительные трудности возникали из-за разного оформления информации о продукте.
    - В результате ~70% фото - найдены все 3 числовых признака, 18% - 2 признака, 10% - 1, 2% - 0.
    - Тип продукта распознавался хорошо практически во всех случаях, кроме ситуаций, когда наименование типа продукта отсутствовало на фото. 
  + Машинное обучение:
    - Среднее значение рекомендуемого количества продукта 189 г
    - MAE: 6.497764998259354
    - MSE: 323.6418901788215
    - RMSE: 17.990049754762257






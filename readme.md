# ml-class-pet-proj


## Используемые библиотеки
*pandas, pyqt, numpy, csv, sklearn.model_selection, zipfile, pyvisa*

# Задача

Имеется макет устройства, осуществляющий одновременное излучение и прием радиосигнала. 
В результате работы макета получаем осциллограмму радиосигнала, отображающую изменение амплитуды сигнала во времени.
Полученная осциллограмма является суммой сигнала прямого излучения и эхо-сигнала, отраженного от цели и окружения вокруг макета. 
Цели могут быть разные, соответственно конфигурации эхо-сигнала тоже будут разными, также цели могут находиться на разном расстоянии от излучателя.

Таким образом задача состоит в том, чтобы научиться распознавать полученные осциллограммы, т.е. понять какой объект находится перед макетом и на каком расстоянии.

Конечной целью данного проекта является создание приложения выполняющего определенный комплекс задач:
 * Приложение должно обладать пользовательским интерфейсом (ПИ);
 * ПИ должен позволять:
   * подключаться к осциллографу по заданному IP адрессу;
   * отображать таблицу классов;
   * редактировать таблицу классов в том числе, добавлять, удалять и редактировать записи;
   * инициализировать съем осциллограмм заданного класса в регулируемом диапазоне;
   * запускать процесс переобучения ML моделей на обновленных данных;
   * запускать процесс предсказания и отображения результата по полученным осциллограммам.
 * Приложение должно удовлетворять все функции ПИ в полном объеме.
 
# Ход решения

Данная задача по своей сути является задачей классификации одномерного ряда.
Было принято решение использовать две независимые модели для классификации класса и расстояния,
 таким образом если система не сможет распознать объект с достаточной долей достоверности, то
  мы хотя бы получим информацию на каком он расстоянии.

Так же применим метод ансамблирования разных моделей из следующих:
 * LogisticRegression
 * RandomForestClassifier
 * lightgbm


## Данные

В результате работы макета имеем одномерный вектор значений амплитуд сигнала по 1250 отсчетам.
Заранее зададимся таблицей классов. Имеем 5 классов объектов, находящихся на 10 разных дальностях и 1 класс с нулевой дальностью, т.е. когда объекта нет перед макетом.

| № | Объект | Класс | Диапазон дальностей|
| :-- | :---------------------- | :---------------------- | :---------------------- |
| 1 | Нет Объекта | 0 | 0 |
| 2 | Объект №1 | 1 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| 3 | Объект №2 | 2 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| 4 | Объект №3 | 3 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| 5 | Объект №4 | 4 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| 6 | Объект №5 | 5 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |

Для устранения дисбаланса обучающей выборки, выбран метод добавления новых экземпляров классов SMOTE.


Структура датасета следующая:
- 0 столбец: класс объекта;
- 1 столбец:  дальность объекта;
- Столбцы с 2 по 1251:  значение амплитуд сигнала по отсчетам.



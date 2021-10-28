import time
import pyvisa as visa
import numpy as np
import os.path

class osci_func:
    def __init__(self, work_path, visa_address, is_pause):
        self.work_path = work_path
        self.visa_address = visa_address
        self.is_pause = is_pause


    def connect(self):


        # Связь с осциллографом
        # visa_address = 'TCPIP::169.254.35.133::INSTR'
        self.rm = visa.ResourceManager()
        self.scope = self.rm.open_resource(self.visa_address)

        self.scope.timeout = 10000
        self.scope.chunk_size = 125000
        print(self.scope.query('*IDN?'))
        print(self.scope.query('*esr?'))
        print(self.scope.query('allev?'))
        rl = self.scope.query('horizontal:mode:recordlength?');
        self.scope.write('DATA:ENCdg ASCii')
        self.scope.write('wfmoutpre:byt_nr 2')
        self.scope.write('data:stop ' + str(rl))
        self.scope.write('data:source MATH1')
        self.yoffset = float(self.scope.query('wfmoutpre:yoff?'))
        self.ymult = float(self.scope.query('wfmoutpre:ymult?'))
        self.yzero = float(self.scope.query('wfmoutpre:yzero?'))
        self.scope.write('data:source MATH1')

    def disconnect(self):
        self.scope.close()
        self.rm.close()

    def predict_run(self):
        # raw_data = np.array(self.scope.query_ascii_values('CURV?'))  # получаем осциллограмму
        # our_data = ((raw_data - self.yoffset) * self.ymult + self.yzero)  # масштабируем
        # while self.is_pause:
        #     time.sleep(1)
        #     print('pause')
        # else:
        #     our_data = np.append(our_data, self.stat_X(our_data))
        #     print(our_data)
        #     return our_data
            # if our_data[20] < 0.05 and our_data[15] > -0.05:  # проверка осциллограмм на корректность
            #     our_data = np.append(our_data, self.stat_X(our_data))
            #     print(our_data)
            #     return our_data

        raw_data = np.array(self.scope.query_ascii_values('CURV?'))  # получаем осциллограмму

        our_data = ((raw_data - self.yoffset) * self.ymult + self.yzero)  # масштабируем
        # our_data = np.append(our_data, self.stat_X(our_data))
        print(our_data.shape)
        return our_data

    def run(self, progress_update, question_signal ,current_class, bottom, up, steps):

        data = np.zeros(1258)  # заготавливаем массив нужной длины
        for i in range(bottom, up + 1):
            question_signal.emit(i)

            while self.is_pause:
                time.sleep(1)

            for j in range(steps):

                time.sleep(0.01)  # таймер между съемом осциллограмм
                raw_data = np.array(self.scope.query_ascii_values('CURV?'))  # получаем осциллограмму

                our_data = ((raw_data - self.yoffset) * self.ymult + self.yzero)  # масштабируем

                if our_data[20] < 0.05 and our_data[15] > -0.05:  #  проверка осциллограмм на корректность
                    our_data = np.append(our_data, self.stat_X(our_data))
                    if current_class == 0:
                        data = np.vstack(
                            (data, np.insert(our_data, 0, [int(current_class), 0])))  # пишем в заготовленный массив данные, класс и дальность
                    else:
                        data = np.vstack(
                            (data, np.insert(our_data, 0, [int(current_class), int(i)])))  # пишем в заготовленный массив данные, класс и дальность

                else:
                    j -= 1
                # print(j)
                progress_update.emit(j + 1)
            self.is_pause = True

        data = data[1:]  # отрезаем инициирующие нули

        path = self.work_path + '/signals'  # путь в папку сохранения
        if os.path.isdir(path):
            pass
        else:
            os.mkdir(path)
        if os.path.isfile(path + '/data.csv'):
            with open(path + '/data.csv', 'a') as f:  # дописываем в файл полученные осциллограммы
                np.savetxt(f, data, delimiter=',')
        else:
            np.savetxt(path + '/data.csv', data, delimiter=',')  # пишем в новый файл



    def stat_X(self, x):
        stat = [np.mean(x), np.median(x), np.std(x), np.var(x), np.max(x), np.min(x)]
        return stat
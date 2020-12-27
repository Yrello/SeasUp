import sys
from predict import pred
from user_statistic import Stat
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtWidgets import QLabel, QLineEdit
from PyQt5.QtGui import QFont


class main(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 330, 290)
        self.setWindowTitle('SeasUp')

        self.btn = QPushButton('Получить', self)
        self.btn.resize(self.btn.sizeHint())
        self.btn.move(20, 180)
        self.btn.clicked.connect(self.get_result)

        self.ageinp = QLineEdit(self)
        self.ageinp.move(110, 10)
        self.ageinp.resize(45, 15)

        self.age = QLabel('• Возраст:', self)
        self.age.setFont(QFont('Tahoma', 10))
        self.age.move(20, 10)

        self.sex = QLabel('• Пол:', self)
        self.sex.setFont(QFont('Tahoma', 10))
        self.sex.move(20, 30)

        self.sexinp = QLineEdit(self)
        self.sexinp.move(110, 30)
        self.sexinp.resize(45, 15)

        self.cigs = QLabel('• Количество сигарет в день:', self)
        self.cigs.setFont(QFont('Tahoma', 10))
        self.cigs.move(20, 50)

        self.cigsinp = QLineEdit(self)
        self.cigsinp.move(200, 50)
        self.cigsinp.resize(45, 15)

        self.chol = QLabel('• Холестерин:', self)
        self.chol.setFont(QFont('Tahoma', 10))
        self.chol.move(20, 70)

        self.cholinp = QLineEdit(self)
        self.cholinp.move(110, 70)
        self.cholinp.resize(45, 15)

        self.bp = QLabel('• Принимаете ли вы препараты влияющие на \n кровяное давление?:', self)
        self.bp.setFont(QFont('Tahoma', 10))
        self.bp.move(20, 90)

        self.bpinp = QLineEdit(self)
        self.bpinp.move(160, 108)
        self.bpinp.resize(45, 15)

        self.glucose = QLabel('• Глюкоза:', self)
        self.glucose.setFont(QFont('Tahoma', 10))
        self.glucose.move(20, 130)

        self.glucoseinp = QLineEdit(self)
        self.glucoseinp.move(110, 131)
        self.glucoseinp.resize(45, 15)

        self.result = QLabel('Результат:', self)
        self.result.setFont(QFont('Tahoma', 10))
        self.result.move(20, 220)

    def get_result(self):
        # user_data = Stat('SeasUp', 0, 0, 0, 0, 0, 0)
        try:
            age = int(self.ageinp.text())
            if self.sexinp.text().lower() == 'м':
                sex = 1
            else:
                sex = 0

            cigs = int(self.cigsinp.text())
            chol = float(self.cholinp.text().replace(',', '.'))

            if self.bpinp.text().lower() == 'да':
                bp = 1
            else:
                bp = 0

            glucose = float(self.glucoseinp.text().replace(',', '.'))
            user_data = Stat('SeasUp', age, sex, cigs, chol, bp, glucose)
            print(user_data.predict_result())
            if user_data.predict_result()[0] == 0:
                self.result.setText(
                    'Результат: С большой вероятностью риск развития\nишемической болезни сердца у вас\nотсутствует')

            elif user_data.predict_result()[0] == 1:
                self.result.setText(
                    'Результат: Предварительный анализ показывает\nчто у вас есть риск развития ишемической\nболезни сердца')
            else:
                pass
        except:
            self.result.setText('Результат: Ошибка')
        self.result.adjustSize()


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    m = main()
    m.show()
    sys.excepthook = except_hook
    sys.exit(app.exec())

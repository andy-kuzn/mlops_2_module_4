MLOps. Практическое задание №4 (vo-HW)

Выбранный инструмент автоматизации процесса машинного обучения: ClearML.
Был развернут сервер ClearML на виртуальной машине Ubuntu в VirtualBox.

Выбранный датасет: Abalone (https://archive.ics.uci.edu/dataset/1/abalone).
Задача по предсказанию возраста (точнее, числа годовых колец) моллюсков Abalone ("морских ушек") по их физическим параметрам (размерам, весу и др.)

Выбранная модель: GradientBoostingRegressor из библиотеки Scikit-learn.

Ноутбуки и скрипты создавались в VS Code, который подсоединялся к репозиторию на виртуальной машине через Openssh server c переадресацией портов.

Эксперементы на сервере ClearML проводились через браузер хостовой машины, также с переадресацией портов.
В ходе экспериментов проводился поиск оптимальных гиперпараметров модели.
В результате удалось несколько улучшить выбранную метрику (MSE) по сравнению со значением, получаемым при дефолтных значениях гиперпараметров.

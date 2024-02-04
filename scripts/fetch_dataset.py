from ucimlrepo import fetch_ucirepo 
  
# Загружаем датасет
abalone = fetch_ucirepo(id=1) 
  
# Выводим признаки и целевую переменную в виде датафреймов
X = abalone.data.features 
y = abalone.data.targets 
  
# Выводим метаданные
print(abalone.metadata) 
  
# Выводим сведения о переменных 
print(abalone.variables) 

# Записываем признаки и целевую переменные
X.to_csv('/home/andrey/projects/abalone/datasets/X_abalone.csv',
         index=None)
y.to_csv('/home/andrey/projects/abalone/datasets/y_abalone.csv',
         index=None)

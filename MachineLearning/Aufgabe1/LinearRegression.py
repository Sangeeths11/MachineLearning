# Sangeeths Chandrakuar
# 2023-09-21
# CDS Machine Learning

from sklearn.linear_model import LinearRegression

# Stunden gelernt und Noten
stunden = [[2], [4], [6], [8]]  # Stunden gelernt
noten = [2.5, 4.0, 5.5, 7.0]   # Erhaltene Noten

# Lineare Regression
modell = LinearRegression().fit(stunden, noten)

# Vorhersage für jemanden, der 5 Stunden gelernt hat
vorhersage = modell.predict([[5]])
print(f"Ein Student, der 5 Stunden gelernt hat, könnte eine Note von {vorhersage[0]:.2f} bekommen.")


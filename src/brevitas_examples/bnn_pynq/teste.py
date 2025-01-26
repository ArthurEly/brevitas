import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Exemplo de rótulos reais e predições
y_true = [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]  # Rótulos reais
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]  # Rótulos previstos

# Gerando a matriz de confusão
cm = confusion_matrix(y_true, y_pred)

# Criando o gráfico da matriz de confusão
disp = ConfusionMatrixDisplay(cm, display_labels=['Classe 0', 'Classe 1'])
disp.plot()

# Salvando o gráfico como um arquivo PDF
plt.savefig("matriz_confusao.svg", format='svg')

# Exibindo o gráfico
plt.show()

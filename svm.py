# Importando bibliotecas necessárias
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
X, y = load_wine(return_X_y=True)

# Testar diferentes kernels, valores de C e gamma
kernels = ['linear', 'rbf', 'poly']
C_values = [0.1, 1, 10]
gamma_values = [0.01, 0.1, 1]

# Dividir o dataset em treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Iterar sobre diferentes combinações de kernel, C e gamma
for kernel in kernels:
    for C in C_values:
        for gamma in gamma_values:
            print("="*30)
            print(f"Treinando SVM com kernel={kernel}, C={C}, gamma={gamma}")
            svm_model = SVC(kernel=kernel, C=C, gamma=gamma)
            svm_model.fit(X_train, y_train)
            
            # Fazendo predições no conjunto de teste
            y_pred = svm_model.predict(X_test)
            
            # Relatórios de classificação
            print(f"\nResultados para kernel={kernel}, C={C}, gamma={gamma}")
            print(classification_report(y_test, y_pred, target_names=load_wine().target_names))
            
            # Matriz de confusão
            cm = confusion_matrix(y_test, y_pred)
            print("Matriz de Confusão:\n", cm)
            
            # Plot da matriz de confusão
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=load_wine().target_names, yticklabels=load_wine().target_names)
            plt.title(f'Matriz de Confusão (kernel={kernel}, C={C}, gamma={gamma})')
            plt.xlabel('Classe Predita')
            plt.ylabel('Classe Verdadeira')
            print("="*30)
            plt.show()

# Justificar o melhor modelo
# Com base nos relatórios de classificação, escolha o kernel, C e gamma que produziu a melhor combinação de precision, recall e f1-score.

# Importar as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Ler a planilha gerada
df = pd.read_excel('caminho_para_sua_planilha.xlsx')  # Substitua pelo caminho da sua planilha

# Pré-processamento dos dados
df['Campanha'] = df['Campanha'].apply(lambda x: 1 if x == 'Sim' else 0)
df['Produto Comprado'] = df['Produto Comprado'].apply(lambda x: 1 if x != 'None' else 0)  # Comprado (1) ou Não Comprado (0)

# Definir variáveis independentes (X) e dependente (y)
X = df[['Histórico de Navegação', 'Campanha']]  # Exemplo de variáveis independentes
y = df['Produto Comprado']  # Variável dependente (se comprou ou não)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Árvore de Decisão
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Previsão e Avaliação para Árvore de Decisão
y_pred_tree = decision_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

# 2. Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Previsão e Avaliação para Random Forest
y_pred_forest = random_forest.predict(X_test)
accuracy_forest = accuracy_score(y_test, y_pred_forest)

# Relatório
relatorio = {
    "Árvore de Decisão - Acurácia": accuracy_tree,
    "Random Forest - Acurácia": accuracy_forest,
}

# Visualizar Árvore de Decisão
plt.figure(figsize=(12, 8))
plot_tree(decision_tree, filled=True, feature_names=X.columns, class_names=['Não Comprado', 'Comprado'], rounded=True)
plt.title("Árvore de Decisão")
plt.show()

# Importância das variáveis no Random Forest
importancia_features = random_forest.feature_importances_

# Mostrar a importância das variáveis
importancia_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importancia_features
}).sort_values(by='Importance', ascending=False)

# Salvar Relatório CSV
relatorio_path = 'relatorio_arvores_decisao_random_forest.csv'
relatorio_df = pd.DataFrame(relatorio, index=[0])
relatorio_df.to_csv(relatorio_path, index=False)

# Mostrar a importância das variáveis no Random Forest
importancia_df_path = 'importancia_features_random_forest.csv'
importancia_df.to_csv(importancia_df_path, index=False)

# Exibir os caminhos dos arquivos gerados
print("Relatório gerado em:", relatorio_path)
print("Importância das variáveis no Random Forest gerada em:", importancia_df_path)
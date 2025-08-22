import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import os

# === Caminho da planilha ===
# Se estiver no mesmo diretório do script, deixe só o nome do arquivo:
caminho_arquivo = r'atividade/comportamento_clientes.xlsx'

# === Leitura ===
df = pd.read_excel(caminho_arquivo)
print("Planilha carregada com sucesso.\n")

print("Primeiras linhas da planilha:")
print(df.head(), "\n")

# === Normalização do alvo ('produto_comprado') => 0/1 ===
alvo_raw = df['produto_comprado'].astype(str).str.strip().str.lower()
positivos = {'1', 'sim', 'yes', 'comprou', 'true'}
negativos = {'0', 'não', 'nao', 'no', 'none', 'nan', '', 'false'}

def map_alvo(x):
    if x in positivos: return 1
    if x in negativos: return 0
    return np.nan

df['produto_comprado'] = alvo_raw.apply(map_alvo)

print("Distribuição de 'produto_comprado' (após normalização):")
print(df['produto_comprado'].value_counts(dropna=False), "\n")

# Validar 2 classes
classes = df['produto_comprado'].dropna().unique()
if len(classes) < 2:
    raise ValueError(
        "A coluna 'produto_comprado' precisa ter pelo menos duas classes (0 e 1). "
        f"No arquivo atual, só encontrei: {classes}.\n"
        "➡️ Adicione linhas com casos de NÃO compra (0)."
    )

# Remover linhas inválidas do alvo
df = df.dropna(subset=['produto_comprado'])
df['produto_comprado'] = df['produto_comprado'].astype(int)

# === campanha como 0/1 ===
camp_raw = df['campanha'].astype(str).str.strip().str.lower()
df['campanha'] = camp_raw.isin({'1', 'sim', 'yes', 'true'}).astype(int)

# === Seleção de features ===
col_numericas = []
if 'Dias desde a última compra' in df.columns:
    col_numericas.append('Dias desde a última compra')

col_binarias = ['campanha']

col_categoricas = []
for col in ['Produto Visualizado', 'Histórico de Navegação']:
    if col in df.columns and df[col].dtype == object:
        col_categoricas.append(col)

# === Montagem do X ===
X = pd.DataFrame(index=df.index)

# Numéricas (coerção + imputação por mediana)
if col_numericas:
    X[col_numericas] = df[col_numericas].apply(pd.to_numeric, errors='coerce')

# Binárias
X[col_binarias] = df[col_binarias]

# One-hot nas categóricas
if col_categoricas:
    dummies = pd.get_dummies(df[col_categoricas], drop_first=True)
    X = pd.concat([X, dummies], axis=1)

# Imputar NaNs numéricos se houver
X = X.fillna(X.median(numeric_only=True))

y = df['produto_comprado']

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Modelo ===
modelo = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced_subsample',
)
modelo.fit(X_train, y_train)

# === Probabilidades ===
idx_pos = list(modelo.classes_).index(1)
y_prob = modelo.predict_proba(X)[:, idx_pos]
df['prob_compra'] = y_prob
df['Probabilidade de Compra'] = (df['prob_compra']*100).map(lambda v: f"{v:.1f}% chance de compra")

df.to_excel('atividade/clientes_com_probabilidades.xlsx', index=False)

# # === Segmentos ===
# alta = df[df['prob_compra'] >= 0.70].copy()
# baixa = df[df['prob_compra'] <= 0.30].copy()

# print("\nClientes com alta probabilidade de compra (>=70%):")
# cols_show = [c for c in ['ID Cliente', 'Probabilidade de Compra'] if c in df.columns]
# print(alta[cols_show].head(20))

# print("\nClientes com baixa probabilidade de compra (<=30%):")
# print(baixa[cols_show].head(20))

# # === Importância das variáveis ===
# importancias = modelo.feature_importances_
# plt.figure(figsize=(8, 6))
# plt.barh(X.columns, importancias)
# plt.title("Importância das Variáveis no Modelo Random Forest")
# plt.xlabel("Importância")
# plt.ylabel("Variáveis")
# plt.tight_layout()
# plt.show()

# # === Exportações ===
# alta.drop(columns=['prob_compra'], errors='ignore').to_csv('clientes_high_probabilidade.csv', index=False)
# baixa.drop(columns=['prob_compra'], errors='ignore').to_csv('clientes_low_probabilidade.csv', index=False)

# planilha_formatada = 'clientes_formatados.xlsx'
# df.drop(columns=['prob_compra'], errors='ignore').to_excel(planilha_formatada, index=False)

# # === Métrica simples ===
# acuracia = modelo.score(X_test, y_test)
# print(f"\nAcurácia do modelo: {acuracia:.2f}")
# print(f"Planilha formatada salva em: {os.path.abspath(planilha_formatada)}")
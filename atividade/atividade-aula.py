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
if 'dias_ultima_compra' in df.columns:
    col_numericas.append('dias_ultima_compra')

col_binarias = ['campanha']

col_categoricas = []
for col in ['produto_visualizado', 'hist_navegacao']:
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

print(X.head())

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
df['probabilidade_compra'] = (df['prob_compra']*100).map(lambda v: f"{v:.1f}% chance de compra")

views = df.groupby('produto_visualizado').size().reset_index(name='qtd_vistos')

# Contar quantas vezes cada produto foi comprado
compras = df[df['produto_comprado'] == 1].groupby('produto_visualizado').size().reset_index(name='qtd_compras')

# Unir os dois dataframes
taxa_produto = pd.merge(views, compras, on='produto_visualizado', how='left')

print(taxa_produto.head())
# Preencher produtos sem compras com 0
taxa_produto['qtd_compras'] = taxa_produto['qtd_compras'].fillna(0)

# Calcular a taxa de compra
taxa_produto['taxa_compra'] = taxa_produto['qtd_compras'] / taxa_produto['qtd_vistos']

df = df.merge(taxa_produto[['produto_visualizado', 'taxa_compra']], on='produto_visualizado', how='left')

df.to_excel('atividade/clientes_com_probabilidades.xlsx', index=False)


# DECIDINDO QUAL PRODUTO DEVE SAIR DE ESTOQUE E QUAL FICAR
# A ideia é usar a taxa de compra para decidir qual produto deve sair de estoque e qual deve ficar.
# Produtos com maior taxa de compra são mais populares e, portanto, devem ser mantidos em estoque.

# Ordenar os produtos pela taxa de compra em ordem decrescente
df_produtos_ordenados = df.sort_values(by='taxa_compra')

produto_sair = df_produtos_ordenados.iloc[0]['produto_visualizado']
print(f"O produto que deve sair de estoque é: {produto_sair} com uma taxa de compra de {df_produtos_ordenados.iloc[0]['taxa_compra']:.2f}")

df_novos_produtos = pd.DataFrame({
    'produto_visualizado': [
        'Nike Zoom Freak 3 (Giannis Antetokounmpo)',
        'Adidas Dame 7 (Damian Lillard)',
        'Jordan Zion 1 (Zion Williamson)',
        'Under Armour Embiid One (Joel Embiid)',
        'Anta KT6 (Klay Thompson)'
    ]
})


print(X.head())
# Copiar média das features dos produtos mais vendidos para criar dados simulados
features_medianas = X.median().to_dict()
print(features_medianas)

# pd.DataFrame([features_medianas]).to_json('atividade/features_medianas.json', indent=4)
X_novos = pd.DataFrame([features_medianas] * len(df_novos_produtos))
X_novos = X_novos.loc[:, X.columns]  # garantir alinhamento de colunas
print(X_novos.head())

# Prever probabilidade de compra dos novos produtos
df_novos_produtos['probabilidade_prevista'] = modelo.predict_proba(X_novos)[:, 1]

print(df_novos_produtos.head())

# Ordenar do mais promissor para o menos
df_novos_produtos = df_novos_produtos.sort_values(by='probabilidade_prevista', ascending=False)
# print(df_novos_produtos.head())





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
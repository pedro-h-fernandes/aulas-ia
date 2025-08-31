alta = df[df['prob_compra'] >= 0.70].copy()
baixa = df[df['prob_compra'] <= 0.30].copy()

print("\nClientes com alta probabilidade de compra (>=70%):")
cols_show = [c for c in ['ID Cliente', 'Probabilidade de Compra'] if c in df.columns]
print(alta[cols_show].head(20))

print("\nClientes com baixa probabilidade de compra (<=30%):")
print(baixa[cols_show].head(20))

# === Importância das variáveis ===
importancias = modelo.feature_importances_
plt.figure(figsize=(8, 6))
plt.barh(X.columns, importancias)
plt.title("Importância das Variáveis no Modelo Random Forest")
plt.xlabel("Importância")
plt.ylabel("Variáveis")
plt.tight_layout()
plt.show()

# === Exportações ===
alta.drop(columns=['prob_compra'], errors='ignore').to_csv('clientes_high_probabilidade.csv', index=False)
baixa.drop(columns=['prob_compra'], errors='ignore').to_csv('clientes_low_probabilidade.csv', index=False)

planilha_formatada = 'clientes_formatados.xlsx'
df.drop(columns=['prob_compra'], errors='ignore').to_excel(planilha_formatada, index=False)

# === Métrica simples ===
acuracia = modelo.score(X_test, y_test)
print(f"\nAcurácia do modelo: {acuracia:.2f}")
print(f"Planilha formatada salva em: {os.path.abspath(planilha_formatada)}")
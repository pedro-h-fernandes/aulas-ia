def calcular_gasto_calorico(sexo, peso, altura, idade, atividade_fisica, objetivo="manutencao"):
    """
    Calcula o gasto calórico diário e ajusta de acordo com o objetivo.

    Args:
        sexo (str): 'masculino' ou 'feminino'.
        peso (float): Peso em quilogramas (kg).
        altura (float): Altura em centímetros (cm).
        idade (int): Idade em anos.
        atividade_fisica (str): 'sedentario', 'leve', 'moderado', 'intenso', 'extremamente_ativo'.
        objetivo (str): 'perda', 'ganho' ou 'manutencao'.

    Returns:
        dict: Contém o gasto calórico e recomendação de calorias para o objetivo.
    """

    # Fórmula de Harris-Benedict revisada
    if sexo.lower() == 'masculino':
        tmb = 66 + (13.7 * peso) + (5 * altura) - (6.8 * idade)
    elif sexo.lower() == 'feminino':
        tmb = 655 + (9.6 * peso) + (1.8 * altura) - (4.7 * idade)
    else:
        return {"erro": "Sexo inválido."}

    fatores_atividade = {
        'sedentario': 1.2,
        'leve': 1.375,
        'moderado': 1.55,
        'intenso': 1.725,
        'extremamente_ativo': 1.9
    }

    fator_atividade = fatores_atividade.get(atividade_fisica.lower())
    if fator_atividade is None:
        return {"erro": "Nível de atividade física inválido."}

    gasto_calorico_total = tmb * fator_atividade

    # Ajuste conforme objetivo
    if objetivo.lower() == "perda":
        recomendacao = gasto_calorico_total - 500  # déficit calórico médio (~0.5 kg/semana)
    elif objetivo.lower() == "ganho":
        recomendacao = gasto_calorico_total + 500  # superávit calórico médio (~0.5 kg/semana)
    elif objetivo.lower() == "manutencao":
        recomendacao = gasto_calorico_total
    else:
        return {"erro": "Objetivo inválido. Use 'perda', 'ganho' ou 'manutencao'."}

    return {
        "tmb": tmb,
        "gasto_calorico_total": gasto_calorico_total,
        "objetivo": objetivo,
        "calorias_recomendadas": recomendacao
    }


# Exemplo de uso
peso_exemplo = 103  # kg
altura_exemplo = 180  # cm
idade_exemplo = 22  # anos
sexo_exemplo = 'masculino'
nivel_atividade_exemplo = 'intenso'
objetivo_exemplo = 'perda'

resultado = calcular_gasto_calorico(sexo_exemplo, peso_exemplo, altura_exemplo, idade_exemplo, nivel_atividade_exemplo, objetivo_exemplo)

if "erro" in resultado:
    print(resultado["erro"])
else:
    print(f"Taxa Metabólica Basal (TMB): {resultado['tmb']:.2f} calorias")
    print(f"Gasto calórico diário estimado (TDEE): {resultado['gasto_calorico_total']:.2f} calorias")
    print(f"Objetivo: {resultado['objetivo'].capitalize()}")
    print(f"Calorias recomendadas para atingir o objetivo: {resultado['calorias_recomendadas']:.2f} calorias")

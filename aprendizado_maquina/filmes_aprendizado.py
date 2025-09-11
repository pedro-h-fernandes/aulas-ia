import pandas as pd
from river import feature_extraction, linear_model, compose
from river import preprocessing
import difflib

# ==========================
# 1. Carregar o dataset
# ==========================
df = pd.read_excel("movies_database.xlsx", engine='openpyxl')
df.fillna("Desconhecido", inplace=True)

# ==========================
# 2. Definir intents e exemplos de perguntas
# ==========================
perguntas = [
    "Qual o filme mais bem avaliado?",
    "Qual o filme com maior bilheteria mundial?",
    "Quem é o diretor de Titanic?",
    "Quantos Oscars ganhou Avatar?",
    "Qual o gênero do filme Interstellar?",
    "Qual o filme com mais indicações ao Oscar?",
    "Qual o filme mais longo?",
    "Qual o filme mais recente?",
]

intencoes = [
    "melhor_nota",
    "maior_bilheteria",
    "diretor",
    "oscar",
    "genero",
    "mais_indicacoes",
    "maior_duracao",
    "filme_recente",
]

# ==========================
# 3. Criar modelo com River
# ==========================
modelo = compose.Pipeline(
    ("vectorizer", feature_extraction.BagOfWords(lowercase=True)),
    ("scale", preprocessing.StandardScaler()),
    ("log_reg", linear_model.LogisticRegression())
)

# Treinar com perguntas e intenções iniciais
for pergunta, intent in zip(perguntas, intencoes):
    modelo = modelo.learn_one({ "pergunta": pergunta }, intent)

# ==========================
# 4. Funções auxiliares
# ==========================
def encontrar_filme(titulo):
    """Retorna o filme mais parecido com o nome fornecido"""
    titulos = df['title'].tolist()
    candidato = difflib.get_close_matches(titulo, titulos, n=1, cutoff=0.6)
    return candidato[0] if candidato else None

def responder(pergunta):
    # Predizer intenção
    intent = modelo.predict_one({"pergunta": pergunta})

    # Respostas baseadas na intenção
    if intent == "melhor_nota":
        filme = df.loc[df['rating_imdb'].idxmax()]
        return f"O filme mais bem avaliado é '{filme['title']}' com nota {filme['rating_imdb']} no IMDb."
    
    elif intent == "maior_bilheteria":
        filme = df.loc[df['gross_world_wide'].idxmax()]
        return f"O filme com maior bilheteria mundial é '{filme['title']}' com ${filme['gross_world_wide']:,} arrecadados."
    
    elif intent == "diretor":
        palavras = pergunta.split()
        for p in palavras:
            filme = encontrar_filme(p)
            if filme:
                diretor = df.loc[df['title'] == filme, 'director'].values[0]
                return f"O diretor de '{filme}' é {diretor}."
        return "Não encontrei o filme na base de dados."
    
    elif intent == "oscar":
        palavras = pergunta.split()
        for p in palavras:
            filme = encontrar_filme(p)
            if filme:
                oscars = df.loc[df['title'] == filme, 'oscar'].values[0]
                return f"O filme '{filme}' ganhou {oscars} Oscars."
        return "Não encontrei informações sobre Oscars para esse filme."
    
    elif intent == "genero":
        palavras = pergunta.split()
        for p in palavras:
            filme = encontrar_filme(p)
            if filme:
                genero = df.loc[df['title'] == filme, 'genre'].values[0]
                return f"O gênero do filme '{filme}' é {genero}."
        return "Não encontrei o gênero do filme."
    
    elif intent == "mais_indicacoes":
        filme = df.loc[df['nomination'].idxmax()]
        return f"O filme com mais indicações ao Oscar é '{filme['title']}' com {filme['nomination']} indicações."
    
    elif intent == "maior_duracao":
        filme = df.loc[df['duration'].idxmax()]
        return f"O filme mais longo é '{filme['title']}' com duração de {filme['duration']} minutos."
    
    elif intent == "filme_recente":
        filme = df.loc[df['year'].idxmax()]
        return f"O filme mais recente é '{filme['title']}', lançado em {filme['year']}."
    
    return "Desculpe, não consegui entender a sua pergunta."

# ==========================
# 5. Loop de interação
# ==========================
print("Bem-vindo ao sistema de filmes! Pergunte sobre os filmes ou digite 'sair' para encerrar.")

while True:
    pergunta_usuario = input("\nSua pergunta: ")

    if pergunta_usuario.lower() == "sair":
        print("Saindo...")
        break

    resposta = responder(pergunta_usuario)
    print(resposta)

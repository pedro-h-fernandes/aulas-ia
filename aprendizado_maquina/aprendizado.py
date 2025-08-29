import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC  # Suporte a máquinas vetoriais
from sklearn.pipeline import make_pipeline

# Definir perguntas e respostas
perguntas = [
    "Qual é o jogo mais popular atualmente?",
    "Qual a melhor motocicleta para iniciantes?",
    "Quais jogos têm o melhor modo multiplayer?",
    "Qual moto tem o melhor desempenho em pistas?",
    "Qual é a moto mais veloz do mundo?",
    "Qual jogo tem a melhor história?",
    "Qual moto o Rogerio tem?"
]

# Respostas associadas às perguntas
# vamos padronizar as perguntas por enquanto, deixar algumas perguntas prontas de forma calculada
respostas = [
    "O jogo mais popular atualmente é League of Legends.",
    "A melhor motocicleta para iniciantes é o patinete elétrico.",
    "Os jogos com o melhor modo multiplayer são [Jogo A] e [Jogo B].",
    "A moto com o melhor desempenho em pistas é a [Moto Z].",
    "A moto mais veloz do mundo é a do Marc Marques.",
    "O jogo com a melhor história é Tomb Raider.",
    "Ele tem uma TDM 850 e uma NC 750x, ambas são bem bacanas."
]

# Intenções associadas a cada pergunta
# aqui podemos separar as intenções por categorias, como genero, ano, bilheteria, qtd de oscares, etc
intencao = [
    "informacao_jogo",
    "informacao_moto",
    "informacao_jogo",
    "informacao_moto",
    "informacao_moto",
    "informacao_jogo",
    "informacao_rogerio"
]

# Criar o modelo de linguagem com TF-IDF e SVM
vectorizer = TfidfVectorizer()
model = make_pipeline(vectorizer, SVC(kernel='linear'))  # Usando SVM com kernel linear

model.fit(perguntas, intencao)  # Treinamento do modelo

# Função para responder perguntas
def responder(pergunta):
    intencao_predita = model.predict([pergunta])[0]  # Predizer a intenção

# aqui podemos manter a aleatoriedade pela categoria da pergunta, ou podemos fazer de forma calculada por assertividade da pergunta de acordo com a estrutura dela

    # Encontrar a resposta correspondente à intenção
    if intencao_predita == "informacao_jogo":
        return random.choice(respostas[:3])  # Respostas sobre jogos
    elif intencao_predita == "informacao_moto":
        return random.choice(respostas[3:6])  # Respostas sobre motos
    elif intencao_predita == "informacao_rogerio":
        return respostas[6]  # Resposta específica sobre o Rogerio
    
# temos que pensar numa forma agora dele conseguir aprender e guardar as novas perguntas e respostas


# Interação com o usuário
while True:
    pergunta_usuario = input("Faça uma pergunta (ou digite 'sair' para terminar): ")

    if pergunta_usuario.lower() == 'sair':
        print("Saindo...")
        break

    resposta_ia = responder(pergunta_usuario)
    print(f"Resposta da IA: {resposta_ia}")
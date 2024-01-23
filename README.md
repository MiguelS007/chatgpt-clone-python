# chatgpt-clone-python
Clone do ChatGPT em Python com LangChain

# Contexto

# O que é Large Language Model?
Uma LLM (Large Language Model) é um modelo de linguagem projetado para gerar texto que se assemelha ao que os seres humanos produzem. Essas LLMs são capazes de entender e responder diversas solicitações, fornecendo respostas relevantes. O ChatGPT é um exemplo de aplicação que utiliza LLMs (GPT-3 e GPT-4), assim como o nosso clone vai utilizar.

# O que é LangChain?
O LangChain é um framework para desenvolvimento de aplicações impulsionadas por modelos de linguagem. Além de chamadas de API, ele permite integrar com banco de dados e muito mais.

# Mão na Massa
Vamos agora por a mão na massa e criar nossa aplicação. Estou assumindo aqui que você já teve um contato prévio com python e tem seu ambiente de desenvolvimento configurado. Recomendo fortemente que seja utilizado uma ferramenta de notebook; como o jupyter, Colab ou Kaggle.

Mas primeiro você precisa de uma API KEY da OpenAI. Essa API KEY permite que a OpenAI nos libere acesso ao seus modelos de LLM. Caso, não tenha uma, pode conseguir ela aqui.

Então vamos desenvolver nosso clone do ChatGPT?

# Importar as bibliotecas necessárias
Primeiro precisamos instalar as dependências, que são as bibliotecas da LangChain e da OpenAI. Então execute o comando abaixo no seu terminal:

```python
pip install langchain openai
```

Agora no seu editor importe as dependências necessárias:

```python
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
```

# Definir a chave da OpenAI
Como mencionado acima, para utilizar os modelos da OpenAI, é necessário uma API KEY. Vamos defini-la abaixo:

# Sua API KEY aqui
```python
openai_api_key = "sk-XXX"
```

# Definir o Template de Prompt
Antes de seguirmos vamos à dois conceitos:

- Prompt: É o texto de entrada fornecido a um modelo de linguagem. Ele guia o modelo na geração de uma resposta.
- Template de Prompt: É uma forma reproduzível de gerar um prompt.

Agora vamos criar nosso template de prompt. Nesse template, vamos instruir o modelo de linguagem a assumir o papel de Jarvis, um assistente pessoal com capacidades semelhantes ao ChatGPT. Para isso utilizaremos a classe PromptTemplate com duas variáveis de entrada: history (para armazenar o histórico da conversa), e human_input (para armazenar a solicitação mais recente ao modelo).

```python
template = """Jarvis é um grande modelo de linguagem treinado pela OpenAI.

O Jarvis foi projetada para ajudar em uma ampla gama de tarefas, desde responder a perguntas simples até fornecer explicações e discussões detalhadas sobre uma ampla variedade de tópicos.
Como um modelo de linguagem, o Jarvis é capaz de gerar texto semelhante ao humano com base na entrada que recebe, permitindo que ele se envolva em conversas com som natural e forneça respostas coerentes e relevantes para o tópico em questão.

O Jarvis está constantemente aprendendo e melhorando, e seus recursos estão em constante evolução.
É capaz de processar e compreender grandes quantidades de texto e pode usar esse conhecimento para fornecer respostas precisas e informativas a uma ampla gama de perguntas.
Além disso, o Jarvis é capaz de gerar seu próprio texto com base nas entradas que recebe, permitindo que ele se envolva em discussões e forneça explicações e descrições sobre uma ampla gama de tópicos.

No geral, o Jarvis é uma ferramenta poderosa que pode ajudar em uma ampla gama de tarefas e fornecer percepções e informações valiosas sobre uma ampla variedade de tópicos.
Se você precisa de ajuda com uma pergunta específica ou apenas deseja conversar sobre um tópico específico, o Jarvis está aqui para ajudar.

{history}
Humano: {human_input}
Jarvis: """

prompt = PromptTemplate(
    input_variables=["history", "human_input"], 
    template=template
)
```

# Definir a Memória
As LLMs, como GPT-3/GPT-4, não possuem memória das interações com o usuário, porém é possível criamos uma aplicação com essa capacidade.

Por exemplo, podemos utilizar o ConversationBufferWindowMemory, do LangChain, que permite que nossa aplicação lembre das interações passadas com o usuário. Ela mantém uma janela deslizante do histórico da conversa. Por exemplo, podemos definir que a aplicação lembre das três ultimas interações:

```python
memory = ConversationBufferWindowMemory(k=3)
```

# Definir a LLM
Nesse passo, vamos definir o Large Language Model a ser utilizado pela nossa aplicação. Como o LangChain podemos utilizar diversos modelos, como BERT, LLaMA, Alpaca, GPT-3 e GPT-4. A ideia aqui é desenvolver um clone do ChatGPT, portanto é interessante utilizamos o modelo da própria OpenAI. Para o GPT-3, temos o modelo gpt-3.5-turbo; e para o GPT-4, temos o modelo gpt-4.

Nesse tutorial, vamos utilizar o gpt-3.5-turbo, mas sinta-se livre para usar o gpt-4, se desejar.

Outra coisa interessante é: nos modelos de chat da OpenAI, temos o parâmetro temperature que controla a aleatoriedade da resposta. Uma valor mais alto nesse parâmetro, como 0.8, faz com que as respostas sejam mais diversas e criativas; enquanto um valor mais baixo, como 0.2, faz com que as respostas sejam mais focadas e determinísticas.

E com isso temos tudo o que é necessário para definir a LLM:

```python
llm = ChatOpenAI(
    temperature=0.8,
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo"
)
```

Agora vamos definir a cadeia LLMChain. Essa cadeia combina tudo que fizemos até agora. Nela, podemos passar um valor de entrada de acordo com o PromptTemplate para obter uma saída do nosso modelo de linguagem.

```python
jarvis_chain = LLMChain(
    llm=llm,
    prompt=prompt, 
    memory=memory,
    verbose=True,
)
```

Repare que definimos a chain no modo verboso, que é interessante para aprendizado, mas não para um ambiente de produção.

# Testar a Chain
Agora vamos podemos desfrutar de todo nosso trabalho. Chegou o momento de utilizar nossa criação.

Para isso basta fazermos uma solicitação. Vamos pedir um filme de comédia com Nicolas Cage.

```python
output = jarvis_chain.predict(
    human_input="me recomende um filme de comédia com nicolas cage?"
)
print(output)

# Saída:
# Uma recomendação de filme de comédia com Nicolas Cage é "Adaptação" (2002).
# Neste filme, Nicolas Cage interpreta dois personagens, os irmãos Charlie e Donald Kaufman, em uma história cômica e original.
# O filme é dirigido por Spike Jonze e tem um elenco talentoso, incluindo Meryl Streep e Chris Cooper.
# "Adaptação" combina humor inteligente com uma trama intrigante e performances excelentes, tornando-o um filme de comédia memorável.
```

```python
output = jarvis_chain.predict(
    human_input="gostei. pode recomendar outro?"
)
print(output)

# Saída:
# Claro! Outra recomendação de filme de comédia com Nicolas Cage é "Raising Arizona" (1987).
# Neste filme dos irmãos Coen, Nicolas Cage interpreta H.I. McDunnough, um ex-detento que se casa com a policial Edwina "Ed" McDunnough, interpretada por Holly Hunter.
# Quando o casal descobre que não pode ter filhos, eles decidem sequestrar um dos quintuplos de um rico empresário local.
# "Raising Arizona" é uma comédia hilariante e excêntrica, cheia de diálogos divertidos e performances cativantes.
# É um filme clássico que certamente vai te fazer rir.
```

```python
output = jarvis_chain.predict(
    human_input="qual é avaliação desse filme pela crítica?"
)
print(output)

# Saída:
# "Raising Arizona" foi bem recebido pela crítica e é considerado um clássico da comédia.
# No site Rotten Tomatoes, o filme possui uma taxa de aprovação de 91% com base em 58 avaliações críticas, com uma média de 8,6/10.
# O consenso crítico afirma: "Raising Arizona provou que os Irmãos Coen estavam um passo à frente do restante de Hollywood desde o início - e estabeleceu Nicolas Cage como um ator icônico".
# O filme também recebeu elogios por seu roteiro inteligente, atuações engraçadas e estilo visual único.
```

Repare que foi lembrado o contexto das conversas aqui, por causa da memória que adicionamos de três interações.

Outro ponto importante: como os modelos GPT-3 e GPT-4 não são determinísticos, muito provavelmente cada vez que forem executadas essas perguntas, teremos respostas diferentes.

# Considerações Finais
Fizemos um exemplo bem simples para demonstrar as capacidades do LangChain com GPT-3/4.

Existe muito potencial nessa combinação, como por exemplo criar um assistente pessoal, gerador de conteúdos para mídias sociais, entre outros.

Como exercício, recomendo expandir essa aplicação e criar algo diferente.
Caso crie algo, sinta-se a vontade pra me marcar nas redes sociais, vou querer ver o que você produziu.

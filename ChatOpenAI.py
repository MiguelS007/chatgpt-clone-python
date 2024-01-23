import os
from dotenv import load_dotenv
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")

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

memory = ConversationBufferWindowMemory(k=3)

llm = ChatOpenAI(
    temperature=0.8,
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo"
)

jarvis_chain = LLMChain(
    llm=llm,
    prompt=prompt, 
    memory=memory,
    verbose=True,
)

output = jarvis_chain.predict(
    human_input="me recomende um filme de comédia com nicolas cage?"
)
print(output)

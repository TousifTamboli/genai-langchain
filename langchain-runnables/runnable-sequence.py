from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke on the {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke {text}',
    input_variables=['text']
)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({'topic':'Narendra Modi'})

print(result)
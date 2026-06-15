from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

def word_counter(text):
    return len(text.split())

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Generate me a Short joke of {topic}',
    input_variables='topic'
)

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_counter)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'Modi'})

print(result)
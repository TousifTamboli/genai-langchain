from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=2048,
    temperature=0.2
)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.2
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text:\n{text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate strictly five short question-answer pairs from the following text:\n{text}, dont give less than five or more than five exactly five",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="""
You are a document formatter.

Combine the following notes and quiz into a single study guide.

Return ONLY the final document.
Do not explain your process.
Do not describe what you are doing.

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    notes=prompt1 | model1 | parser,
    quiz=prompt2 | model2 | parser
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Because the term "transformers" spans multiple unrelated fields, the most relevant details are split below based on what you might be searching for: Artificial Intelligence (Deep Learning), Electrical Engineering, or the Entertainment Franchise. [1, 2, 3, 4] 
------------------------------
## 1. Transformer Models (Artificial Intelligence & Deep Learning)
In AI, a transformer is a dominant neural network architecture used to process sequential data. Introduced by Google researchers in the seminal 2017 paper ["Attention Is All You Need"](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf), it serves as the foundation for modern Generative AI like OpenAI's ChatGPT, Google's Gemini, and Meta's Llama. [5, 6, 7, 8] 

* Self-Attention Mechanism: Unlike older models (like RNNs) that process text word-by-word, transformers look at an entire sentence at once. Self-attention mathematically calculates how strongly different words in a sequence relate to each other, allowing the model to capture deep context. [2, 5, 6, 9, 10] 
* Parallel Processing: Because it processes data simultaneously rather than sequentially, transformers can be trained exponentially faster on massive datasets using modern GPUs. [2, 11, 12, 13, 14] 
* Key Components:
* Tokens & Embeddings: Text is broken down into small units (tokens) and converted into numerical vectors.
   * Positional Encoding: Since words are processed all at once, the model injects distinct mathematical markers to remember the original order of the words.
   * Encoder-Decoder Architecture: The encoder digests the input sequence contextually, and the decoder generates the output sequence (e.g., translating languages or predicting the next token). [2, 8, 9, 15, 16] 

------------------------------
## 2. Electrical Transformers (Engineering & Power Grids)
In electrical engineering, a transformer is a passive, static device that transfers electrical energy between circuits through electromagnetic induction. Its primary job is to alter alternating current (AC) voltage levels without changing the frequency of the power supply. [1, 17, 18] 

* Working Principle: It relies on Faraday's Law of Induction. When AC power flows through an input (primary) coil of wire, it creates a fluctuating magnetic field in a shared iron core. This shifting magnetic field induces an electromotive force (voltage) across an output (secondary) coil. [1, 19, 20] 
* Step-Up vs. Step-Down:
* Step-Up: Has more wire turns on the secondary coil than the primary. It increases voltage (and reduces current) to transmit electricity efficiently over long-distance power lines with minimal energy loss.
   * Step-Down: Has fewer wire turns on the secondary coil. It decreases voltage to safe, usable levels for household appliances and commercial buildings. [20, 21, 22, 23, 24] 
* Core Design: The central iron core is constructed from thin, laminated sheets of steel. This lamination is vital to minimize power loss caused by eddy currents (internal swirling electrical currents that generate waste heat). [25, 26, 27] 

------------------------------
## 3. Transformers Franchise (Entertainment)
If you are looking for pop culture information, Transformers is a massive global media franchise owned by Hasbro and Takara Tomy. [28, 29, 30, 31] 

* The Lore: It centers on two warring factions of sentient, living robotic alien beings—the heroic Autobots (led by Optimus Prime) and the evil Decepticons (led by Megatron). They originate from the planet Cybertron and possess the ability to "transform" their bodies into vehicles, weapons, and biomimetic creatures. [32, 33, 34, 35, 36] 
* Media History: Beginning as a toy line in 1984, the franchise spans popular 1980s comic books, iconic animated series, and a multi-billion-dollar live-action cinematic universe spearheaded by director Michael Bay. [32, 37, 38, 39, 40] 

------------------------------
"""

result = chain.invoke({"text": text})

print("\n" + "=" * 80)
print("FINAL RESULT")
print("=" * 80)

print(result)

print("\n" + "=" * 80)
print("CHAIN GRAPH")
print("=" * 80)

chain.get_graph().print_ascii()
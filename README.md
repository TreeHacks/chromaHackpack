# Chroma Hackpack

# Overview

Chroma is a vector database. In this hackpack, we'll use it to implement
retrieval augmented generation (RAG) – a technique for enhancing large language
model (LLM) informational capabilities. Specifically, we'll build a chat bot to
answer logistic questions about TreeHacks 2024.

## Motivation

LLMs like ChatGPT are capable
of solving sophisticated tasks. However, their knowledge of current events and
new information is often limited by training cut-off dates. Moreover, LLMs can
exhibit hallucinatory behavior. In other words, LLMs have strong reasoning
abilities but they often need the appropriate facts to reason with.

Rerieval augmentented generation is a technique of **retrieving** information and
then providing it to the LLM to **augment** the content it next **generates**. 
This helps mitigate hallucination and supplements the LLM's existing knowledge
with facts of the developer's choice.

In the case of our application, we'll use RAG to provide our chat bot up-to-date
information regarding TreeHacks 2024.

## How does RAG work?

Typically, LLMs directly respond to a user's query. Retrievel augmented
generation modifies the query by including relevant facts to the query.

First, we select several documents containing information relevant to TreeHacks
logistics. We then calculate embeddings for the document contents. Embeddings
are vectors that represent the semantics of a given string. If two vector
embeddings are similar, then we know the semantics of the two respective strings
are also similar. These embeddings are all loaded into our vector database,
Chroma. 

Once our vector database is populated, we can begin querying it. When the user
prompts our chat bot, the following occurs:

(1) Take in user input.
(2) Pass the input's embedding into a vector database. Retrieve the `k` most
similar vectors and their associated strings. Each of these strings represent
the information that is most relevant to the user's query.
(3) Pass the user's original input along with the information from the vector
database into the LLM.
(4) Return the LLM's output.

![RAG with Chroma diagram.](https://docs.trychroma.com/img/hrm4.svg)

This framework is simple, but powerful. There are several ways to introduce
additional sophistication into RAG, but for the purpose of this hackpack we'll
focus on the basics.

# Project Walkthrough

## Step 0: Installing Dependencies

Ensure you have Python 3+ installed on your computer. 

Download this repository and run `pip install -r requirements.txt`.

## Step 1: Setting up Chroma

Before we can process user queries, we must populate a Chroma vector database
with embeddings.

### Pre-processing

We will use the `langchain` library to load and pre-process our data.

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

First, we'll use the `DirectoryLoader` to load all the files from our
`documents` folder. Then, we'll use the `RecursiveCharacterTextSplitter` to
break each document down into a series of strings. Each string will have its own
embedding and thus can be independently queried.

```python
loader = DirectoryLoader('./documents')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
```

You'll notice that the `documents` folder comes pre-populated with TreeHacks
related documents. You may replace these documents if you'd like your RAG LLM to be
fed a different set of information. 

### Loading into Chroma

Setting up our Chroma database is very easy.

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
```

We will use `OpenAIEmbeddings` to embed our texts. However, we don't need to
manually do this – Chroma will handle it. We simply declare our Chroma database
with the texts and the embedding function.

```python
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory='db')
```

This will automatically produce a Chroma vector database containing all the text
documents and their vector embeddings. If you'd like to use a [different
embedding function](https://python.langchain.com/docs/integrations/text_embedding), you can easily replace it.

Before running this code, you will need to set up your API key. Use this
[tutorial](https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key)
provided by OpenAI.

Moreover, notice that we pass a value for `persist_directory`. This tells Chroma
to locally save the vector database to the folder `db`. By doing so, we can
simply load in the vector database next time we run our program. This allows us
to avoid recomputing the embeddings for all the documents.

## Step 2: Running Queries

Now that we've configured our Chroma database, we'd like to query it for the
purpose of RAG. `langchain` gives us a pre-packaged object to do this.

```python
from langchain_openai import OpenAI
from langchain.chains import VectorDBQA
```

The `VectorDBQA` automatically coordinates interactions between our LLM and
Chroma vector database. We can declare it easily.

```python
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)
```

To ask it a question, we can simply call the `invoke` method.

```python
query = "What are the top prize categories?"
out = qa.invoke(query)
print(out)
```

This outputs: `The hackathon starts on Friday, February 16th at 9pm.`

(You may notice a slightly different output when running this script, since ChatGPT is
non-deterministic. It should be of similar content. however.)

## Step 3 (optional): User Interface

We have produced a minimally viable instance of RAG! However, most users are
probably in want of a more friendly user interface for development or usage
purposes. 

To achieve this, we can use the `gradio` library.

```python
import gradio as gr
```

Gradio gives us a convenient chatbot template we can simply define some logic
for. Let us first declare our chatbot response function.

```python
def response(message, history):
    h = ''

    for d in history:
        h += 'User message: \'' + d[0] + '\', '
        h += 'Bot message: \'' + d[1] + '\' \n'

    m = 'You are an chatbot meant to answer participant questions about TreeHacks, a hackathon. Here is the prior message history: \n' + h + '\nHere is the message you have just been given: ' + message
    yield qa.run(m)
```

This function accepts two variables: the most recent message from the user and a
history of previous messages. We format the chat history into a single string
such that our chatbot is always aware of the conversation's whole context.
Although re-formatting this string every function call is certainly not the most
elegant or efficient approach, it will suffice for our proof-of-concept.

Notice that we also use this formatting step to provide additional context
regarding the chatbot's purpose. This is a simple technique for focusing the
chatbot's responses.

To start our user interface, we can run the following line.

```python
gr.ChatInterface(response).launch()
```

You should see a local URL printed in the terminal. Use this to access the gradio chat
interface.

# Thanks
Thanks to everyone at Chroma for supporting this hackpack and TreeHacks!

# Additional Resources
- This hackpack is heavily derived from Harrison Chase's
  [chroma-langchain](https://github.com/hwchase17/chroma-langchain) demo repo.
  Please check it out!
- Chroma has a variety of integratons and features, including multi-modal
  capabilities. Check out their [documentation](https://docs.trychroma.com/) to
  learn more.

RAG Systems and the Power of Self Querying: A Simple Guide

If you’re diving into the world of Retrieval Augmented Generation (RAG) systems, you might have stumbled upon a common conundrum of many faces: When and where to employ semantic search? Don’t worry; by the end of this article, you’ll have a clearer perspective on this!

Before we proceed, let’s stay connected! Please consider following me on Medium, and don’t forget to connect with me on LinkedIn for a regular dose of data science and deep learning insights.” 🚀📊🤖

Semantic Search: What is it?

Let’s kick off with the basics. Semantic search is all about understanding a user’s intent rather than just analyzing the keywords. It digs deeper into the context behind a query, trying to provide a more refined search result.

The Misconception with RAG and Semantic Search

Now, here’s where many go wrong. There’s an overwhelming inclination to use semantic search for, well, everything. Here’s a nugget of wisdom: Not every query needs the weight of semantic search.

Imagine going to a library. For some books, you’d know the exact shelf and position. For others, you might have a vague idea and need assistance. The former is akin to a direct database search — quick and straightforward. The latter is where semantic search shines, helping you understand and find content based on the deeper meaning.

When to Use Semantic Search?

Complex Text Analysis: If your task involves extracting deeper meaning or understanding context, semantic search is your go-to.
Unstructured Data: Let’s say you’re dealing with a massive amount of text data without any clear organization. Here, semantic search can help by understanding the context and providing relevant results.
Not for Simple Look-ups: Remember, if your data involves basic integers or simple strings, stick with the traditional search. It’s like using a calculator to add 2 + 2 — overkill!
The Rise of Self-Querying Retrieval

The solution might lie in the concept of self-querying retrieval. Imagine a system smart enough to decide when to employ semantic search or when a basic search would suffice. Such systems can dynamically alter their search approach based on the query type, leading to more efficient and accurate results.


visual representation of the self-querying
The visual representation of the self-querying process offers a window into its intricate yet efficient operation. Imagine a flowchart, where the user’s query sets the process in motion.

Initial User Query: This is where it all begins. A user types in a search request, looking for specific data.
Refinement via Language Model: Enter the role of a sophisticated large language model. This model processes the query to grasp its semantic nuances. It then optimizes the query for effective data retrieval by aligning it with existing metadata.
Why is this Step Vital?

Consider the movie search scenario mentioned earlier. If you’re searching for a film from a particular year, an optimized system won’t venture deep into the realms of semantic search to determine the year. Instead, it directly fetches movies from that specific year. The keyword here is efficiency; direct lookup for specific data versus deep semantic processing when broader context is needed.

The Spotify Example: Deciphering the Artist Query

Spotify, the popular music streaming platform, offers another stellar example. Say you’re on the hunt for songs by a specific artist. The logical and efficient approach would be to conduct a direct search using the artist’s name, rather than delving into the semantic meaning behind their name. After all, “Adele” is not just a word; it represents a globally acclaimed artist.

But here’s where the magic of semantic search comes into play: if the query is more ambiguous, like “songs that make me feel like dancing under the rain,” the system would utilize semantic search. This type of search can comprehend the mood, emotions, and context to return songs that align with the sentiment.

Striking the Balance

The crux of self-querying lies in understanding which tool to use when. It’s about knowing when to pick the hammer and when the screwdriver is the better choice.

Direct Lookup: Ideal for specific, straightforward requests. Like seeking an artist’s tracks or looking for a movie from 1994.
Semantic Search: This shines when handling ambiguous or context-rich queries that demand a deeper understanding to generate relevant results.
Code Time
To illustrate this concept, let’s delve into some code that employs LangChain’s self-querying retriever, with OpenAI embeddings.

import os

import pinecone


pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"])
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


embeddings = OpenAIEmbeddings()
# create new index
pinecone.create_index("langchain-self-retriever-demo", dimension=1536)
LangChain comes with a built-in self-querying retriever. The key step is specifying the metadata info corresponding to each document attribute.

docs = [
    Document(page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose", metadata={"year": 1993, "rating": 7.7, "genre": ["action", "science fiction"]}),
    Document(page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...", metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2}),
    Document(page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea", metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6}),
    Document(page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them", metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3}),
    Document(page_content="Toys come alive and have a blast doing so", metadata={"year": 1995, "genre": "animated"}),
    Document(page_content="Three men walk into the Zone, three men walk out of the Zone", metadata={"year": 1979, "rating": 9.9, "director": "Andrei Tarkovsky", "genre": ["science fiction", "thriller"], "rating": 9.9})
]
After defining the metadata, the language model (in our case, OpenAI’s model) is initialized, followed by the setup for the retriever.

Preparing the Data
Our dataset for this example comprises a collection of movies, each possessing attributes like genre, year, director. The semantic search will primarily focus on the description, while metadata attributes like year and genre facilitate direct lookups.

metadata_field_info=[
    AttributeInfo(
        name="genre",
        description="The genre of the movie", 
        type="string or list[string]", 
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released", 
        type="integer", 
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director", 
        type="string", 
    ),
    AttributeInfo(
        name="rating",
        description="A 1-10 rating for the movie",
        type="float"
    ),
]
After defining our dataset, it’s embedded using pipecone as the vector store and then stored in a Pinecone DB.

vectorstore = Pinecone.from_documents(
    docs, embeddings, index_name="langchain-self-retriever-demo"
)

document_content_description = "Brief summary of a movie"
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(llm, vectorstore, document_content_description, metadata_field_info, verbose=True)
Making Queries
With the retriever in place, querying becomes intuitive. For example, a query like “What are some movies about dinosaurs?” would return dinosaur.

retriever.get_relevant_documents("What are some movies about dinosaurs")
    query='dinosaur' filter=None


    [Document(page_content='A bunch of scientists bring back dinosaurs and mayhem breaks loose', metadata={'genre': ['action', 'science fiction'], 'rating': 7.7, 'year': 1993.0}),
     Document(page_content='Toys come alive and have a blast doing so', metadata={'genre': 'animated', 'year': 1995.0}),
     Document(page_content='A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea', metadata={'director': 'Satoshi Kon', 'rating': 8.6, 'year': 2006.0}),
     Document(page_content='Leo DiCaprio gets lost in a dream within a dream within a dream within a ...', metadata={'director': 'Christopher Nolan', 'rating': 8.2, 'year': 2010.0})]
Asking for “What’s a movie after 1990 but before 2005 that’s all about toys, and preferably is animated” would entail both a semantic search for “toys” and a metadata-based range filter on the year.

# This example specifies a query and composite filter
retriever.get_relevant_documents("What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated")
    query='toys' filter=Operation(operator=<Operator.AND: 'and'>, arguments=[Comparison(comparator=<Comparator.GT: 'gt'>, attribute='year', value=1990.0), Comparison(comparator=<Comparator.LT: 'lt'>, attribute='year', value=2005.0), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='genre', value='animated')])


    [Document(page_content='Toys come alive and have a blast doing so', metadata={'genre': 'animated', 'year': 1995.0})]
Flexibility and Tolerance
An advantage of using a robust language model is its ability to interpret queries, even if they contain minor errors like misspellings or improper capitalization.

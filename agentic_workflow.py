import os
import openai
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
import torch
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
import logging
import asyncio
from hashlib import blake2b
import getpass
Load environment variables
load_dotenv(dotenv_path="development.env")

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure Qdrant Client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    verify=False
)

COLLECTION_NAME = "my_collection"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("Agents")

# Load the embedding model
MODEL_NAME = "intfloat/e5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=False)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, token=False)

def embed_query(query: str) -> list:
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
import asyncio

def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:  # No event loop exists in the current thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)



class OpenAI_LLM_Service:
    """
    A class to handle a connection with OpenAI's API service.

    This class provides methods to authenticate, send requests, and handle responses
    from the OpenAI API service.
    """

    def __init__(self):
        """
        Initializes the OpenAI_LLM_Service object with the provided API key.
        """
        #self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_key = "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"
        if not self.api_key:
            raise ValueError("Missing required OpenAI API key in environment variables.")

        # Set up the API configuration globally
        openai.api_key = self.api_key
        LOGGER.debug("OpenAI LLM is initiated")

    def generate_simple_response(self, messages: list) -> str:
        """
        Generates a response from the OpenAI service.

        Args:
            messages (list): A list of message dictionaries for the chat model.

        Returns:
            str: The response message content.
        """
        formatted_messages = [{"role": "user", "content": msg} for msg in messages]
        LOGGER.debug(f"Formatted messages for OpenAI: {formatted_messages}")

        response = openai.ChatCompletion.create(
            model="gpt-4",  # Specify the model
            messages=formatted_messages  # Format for ChatCompletion
        )
        return response.choices[0].message.content

    def generate_response(self, system_prompt: dict, messages: list) -> dict:
        """
        Generates a response from the OpenAI service with a system prompt.

        Args:
            system_prompt (dict): The system prompt for context.
            messages (list): A list of user messages.

        Returns:
            dict: The assistant's response.
        """
        # Flatten the list of messages
        formatted_messages = [system_prompt] + [
            {
                "role": "user",
                "content": msg.content if hasattr(msg, 'content') else str(msg)
            }
            for msg in messages
        ]
        LOGGER.debug(f"Formatted messages for OpenAI: {formatted_messages}")

        # Send the request to the OpenAI service
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=formatted_messages
        )
        return {"role": "assistant", "content": response.choices[0].message.content}

    async def generate_response_async(self, system_prompt: dict, messages: list) -> dict:
        """
        Asynchronously generates a response using the synchronous generate_response method.

        Args:
            system_prompt (dict): The system prompt for context.
            messages (list): A list of user messages.

        Returns:
            dict: The assistant's response.
        """
        # Prepare the formatted messages
        formatted_messages = [system_prompt] + [
            {"role": "user", "content": msg.content if hasattr(msg, "content") else str(msg)}
            for msg in messages
        ]
        # Run the synchronous generate_response method in a separate thread
        return await asyncio.to_thread(self.generate_response, system_prompt, formatted_messages)

    def get_embedding(self, text: str) -> list:
        """
        Generates an embedding for the given text using the OpenAI embedding model.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            list: The embedding vector for the input text.
        """
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response['data'][0]['embedding']

    def vector_search(self, query: str, top_k: int = 5) -> list:
        """
        Performs a vector-based search using Qdrant and the embedding generated for the query.

        Args:
            query (str): The search query text.
            top_k (int): The number of top results to retrieve.

        Returns:
            list: A list of search results containing the matched documents.
        """
        # Generate an embedding for the query text
        embedding = embed_query(query)

        # Perform a vector-based search using Qdrant
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=top_k,
            with_payload=True,
        )
        # Extract and return relevant data from the results
        return [
            {
                "source": result.payload.get("source"),
                "content": result.payload.get("chunk"),
                "score": result.score
            }
            for result in results
        ]

class State(TypedDict):
    messages: Annotated[list, add_messages]
    classification: str
    file_names: list
    assistant_response: dict



class Conditions:
    def __init__(self):
        """
        Initializes the Conditions class.
        """
        pass

    @staticmethod
    def decide_agent_based_on_class(state: State) -> str:
        """
        Determines the next node in the graph based on the classification provided in the state.

        Args:
            state (State): The current state containing the classification.

        Returns:
            str: The next node in the graph to transition to.
        """
        classification = state.get("classification")
        LOGGER.debug(f"Within decide_agent_based_on_class, the classification is {classification}")

        if classification == "Summarize":
            LOGGER.debug("Summarizer agent is selected.")
            return "summarizer_agent"
        elif classification == "Complex_question":
            LOGGER.debug("Multihop agent is selected.")
            return "multihop_agent"
        else:
            LOGGER.debug("End of graph reached.")
            return END

class Tools:
    def __init__(self):
        self.limit_length = 10000  # Limit for text processing

    @staticmethod
    def encode_filename(file_name: str) -> str:
        """
        Generate a unique hash for a given file name.
        """
        hash_val = blake2b(digest_size=16)
        hash_val.update(file_name.encode("utf-8"))
        h_as_str = hash_val.hexdigest()
        return f"{h_as_str[0:8]}-{h_as_str[8:16]}-{h_as_str[16:24]}-{h_as_str[24:32]}"

    async def summarize_content(self, text: str) -> str:
        """
        Summarizes the provided text using OpenAI's API.
        """
        text = text[:self.limit_length]  # Trim the text to fit the limit
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the following text in a concise paragraph."},
                {"role": "user", "content": text}
            ]
        )
        return response['choices'][0]['message']['content']

    async def summarize_tool(self, file_names: list) -> dict:
        """
        Summarizes content of multiple files.
        """
        summaries = {}
        for file_name in file_names:
            try:
                with open(file_name, "r") as file:
                    content = file.read()
                    summary = await self.summarize_content(content)
                    summaries[file_name] = summary
            except Exception as e:
                summaries[file_name] = f"Error reading or summarizing file: {e}"
        return summaries

    def vector_search_tool(self, question: str, top_k: int = 5) -> str:
        """
        Performs a vector search in Qdrant for the given question.
        """
        embedding = embed_query(question)  # Generate embedding for the query
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=top_k,
            with_payload=True,
        )
        if not results:
            return "No relevant results found."
        
        # Format the retrieved results
        context = "\n\n".join(
            f"File: {result.payload.get('source', 'Unknown Source')}\n"
            f"Content: {result.payload.get('chunk', 'No content available')}"
            for result in results
        )
        return context

    async def vector_search_tool_async(self, question: str) -> str:
        """
        Asynchronous version of vector_search_tool.
        """
        return await asyncio.to_thread(self.vector_search_tool, question)

    @staticmethod
    def get_prompt_for_answer_generator() -> dict:
        """
        Defines a system message prompt for generating responses.
        """
        return {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the information retrieved from the internal database to generate an accurate and relevant answer to the given question. "
                "Your goal is to use the retrieved information effectively, even if partial, to provide the most informative response possible. "
                "If the retrieved information is insufficient to fully answer the question, clearly indicate this, but still try to provide a partial answer based on the available data, if relevant. "
                "Avoid including information that is not based on the retrieved data.\n\n"
                "### Examples:\n"
                "#### Example 1:\n"
                "Question: What are Covestro's main products?\n"
                "Retrieved Information: Covestro produces polycarbonates (PCS), polyurethanes (PUR), and 4,4'-methylenediisocyanate (MDI).\n"
                "Response: Covestro produces polycarbonates (PCS), polyurethanes (PUR), and 4,4'-methylenediisocyanate (MDI).\n\n"
                "#### Example 2:\n"
                "Question: How does Covestro reduce its environmental impact?\n"
                "Retrieved Information: Covestro has implemented energy-efficient technologies and utilizes renewable resources for chemical production.\n"
                "Response: Covestro reduces its environmental impact by implementing energy-efficient technologies and utilizing renewable resources for chemical production.\n\n"
                "#### Example 3:\n"
                "Question: What are Covestro's strategies for increasing sales in Asia?\n"
                "Retrieved Information: Covestro has expanded its production capacity in China.\n"
                "Response: Covestro is increasing sales in Asia by expanding its production capacity in China.\n\n"
                "#### Example 4:\n"
                "Question: What are Covestro's strategies for increasing sales in Europe?\n"
                "Retrieved Information: [No relevant data retrieved from the internal database]\n"
                "Response: The internal database does not contain sufficient information to fully answer the question: What are Covestro's strategies for increasing sales in Europe? No relevant data was retrieved.\n\n"
                "Now, based on the retrieved information, generate an appropriate response to the question."
            )
        }

    def answer_generator_tool(self, question: str, context: str) -> str:
        """
        Generates an answer using the provided question and context.
        """
        system_prompt = self.get_prompt_for_answer_generator()
        messages = [
            {"role": "system", "content": system_prompt["content"]},
            {"role": "user", "content": f"Query: {question}\n\nContext:\n{context}"}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response['choices'][0]['message']['content']

    async def answer_generator_tool_async(self, question: str, context: str) -> str:
        """
        Asynchronous version of the answer generator tool.
        """
        return await asyncio.to_thread(self.answer_generator_tool, question, context)

    def deconstruct_question_tool(self, question: str) -> list:
        """
        Decomposes a complex question into simpler sub-questions.
        """
        system_prompt = {
            "role": "system",
            "content": (
                "You are a knowledgeable assistant specializing in the chemical industry. Your task is to break down a complex question into standalone, specific, and self-contained sub-questions. "
                "Each sub-question must include enough detail from the original query to make sense independently and be answerable without relying on the other sub-questions.\n\n"
                "### Guidelines:\n"
                "1. Rewrite each sub-question to stand on its own, avoiding vague phrases such as 'these chemicals' or 'this process.' Include explicit details for clarity.\n"
                "2. Focus on dividing the query into core areas, such as products, processes, markets, applications, challenges, or environmental considerations.\n"
                "3. Avoid making assumptions about specific chemicals, products, or regions unless explicitly mentioned in the original query.\n"
                "4. Limit the number of sub-questions to 3, and ensure they are meaningful and actionable.\n\n"
                "### Examples of Complex Questions and Sub-Questions:\n"
                "- Complex Question: How does a company produce its chemicals, and where are they sold?\n"
                "  1. What are the main chemicals produced by the company?\n"
                "  2. What are the chemical processes used by the company for production?\n"
                "  3. Where are the company's chemical products marketed or sold?\n\n"
                "- Complex Question: What are the environmental challenges associated with chemical manufacturing, and how can they be addressed?\n"
                "  1. What are the key environmental challenges in chemical manufacturing?\n"
                "  2. What are the strategies for reducing the environmental impact of chemical production?\n"
                "  3. What are the regulations or best practices related to sustainability in chemical manufacturing?\n\n"
                "- Complex Question: What are the key applications of a chemical product in various industries, and how is it synthesized?\n"
                "  1. What are the main industrial applications of the chemical product?\n"
                "  2. What are the chemical synthesis methods used to produce the product?\n"
                "- Complex Question: What are the key steps in producing a polymer and its industrial applications?\n"
                "  1. What are the primary steps involved in the production of the polymer?\n"
                "  2. What raw materials are required for polymer production?\n"
                "  3. What are the main industries that utilize the polymer?\n\n"
                "Always ensure that sub-questions are clear, detailed, and self-contained. Do not assume prior knowledge. Here is the complex question: "
            )
        }
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt["content"]},
                {"role": "user", "content": question}
            ]
        )
        sub_questions = response['choices'][0]['message']['content'].split("\n")
        return [q.strip() for q in sub_questions if q.strip()]

    def generate_final_answer(self, original_question: str, responses: list) -> str:
        """
        Combines multiple partial responses into a final comprehensive answer.
        """
        system_prompt = {
            "role": "system",
            "content": (
                "You are a helpful assistant. Your task is to consolidate multiple partial answers "
                "to provide a coherent, precise, and complete response to the user's original question. "
                "Use the provided partial answers as context and focus on answering the original question comprehensively. "
                "Avoid repeating similar information and ensure the response is clear and relevant to the user's intent.\n\n"
                "### Examples:\n"
                "Original Question: How does Covestro manufacture its chemicals, and where are they sold?\n"
                "Sub-Answers:\n"
                "1. Covestro produces polycarbonates (PCS) and polyurethanes (PUR) using phosgene-based processes.\n"
                "2. Covestro sells PCS in Antwerp and Caojing.\n"
                "Final Answer: Covestro manufactures its chemicals, including polycarbonates (PCS) and polyurethanes (PUR), "
                "using phosgene-based processes. These chemicals are sold in Antwerp and Caojing.\n\n"
                "Now consolidate the following partial answers to answer the original question:"
            )
        }
        messages = [
            {"role": "system", "content": system_prompt["content"]},
            {"role": "user", "content": f"Original Question: {original_question}\n\nPartial Answers:\n" + "\n".join(responses)}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response['choices'][0]['message']['content']


class Agents:
    
    def __init__(self, llm_service: OpenAI_LLM_Service):
        """
        Initializes the Agents class with the provided LLM service.
        """
        self.llm_service = llm_service  # Store the LLM service instance
        self.tools = Tools()  # Initialize Tools with no dependency on LLM for now

    @staticmethod
    def get_prompt_for_classifier_agent() -> dict:
        return {
            "role": "system",
                "content": "You are a helpful assistant. Your task is to undersand the sentiment behind a user query"
                        " and classify it into 5 classes. The user is an employee of Covestro which is a chemical"
                        " producing company. It has products like polyeurethane and polycarbonates among others."
                        "Here are the 5 classes:"
                        "1. Direct_answer: When the query is very general and can be answered by the LLM independently without"
                        " the need to retrieve any internal knowledge from Covestro's database.\n"
                        "2. Summarize: If the user is requesting to summarize a document or multiple documents.\n"
                        "3. Simple_question: The answer to the user's query requires Covestro's internal database."
                        " It is a simple question that does not need to be deconstructed.\n"
                        "4. Complex_question: The answer to the user's query requires Covestro's internal database."
                        " Deconstructing the query into simpler questions would improve the quality of the answer.\n"
                        "5. Unclear_intent: When the user submits a poorly formulated query, vague phrases, or words"
                        " that do not clearly express an intent.\n\n"
                        "### Examples:\n"
                        "User Query: What is the boiling point of water?\n"
                        "Direct_answer\n\n"
                        "User Query: How many planets are in the solar system?\n"
                        "Direct_answer\n\n"
                        "User Query: Hi\n"
                        "Direct_answer\n\n"
                        "User Query: Can you summarize this report on sustainable materials?\n"
                        "Summarize\n\n"
                        "User Query: Provide a summary of these meeting notes on renewable energy projects.\n"
                        "Summarize\n\n"
                        "User Query: What are the main products of Covestro?\n"
                        "Simple_question\n\n"
                        "User Query: How does Covestro produce polycarbonates?\n"
                        "Simple_question\n\n"
                        "User Query: How can Covestro improve the efficiency of its polyurethane production process?\n"
                        "Complex_question\n\n"
                        "User Query: What strategies can Covestro use to minimize its environmental impact while increasing sales?\n"
                        "Complex_question\n\n"
                        "User Query: Materials\n"
                        "Unclear_intent\n\n"
                        "User Query: Polyeurethane chemistry\n"
                        "Unclear_intent\n\n"
                        "User Query: Research report\n"
        }

    def classifier_agent(self, state: State) -> State:
        LOGGER.debug("Classifier agent triggered")
        system_prompt = self.get_prompt_for_classifier_agent()
        # Correcting the way we access the content of HumanMessage objects
        messages = [{"role": "user", "content": msg.content} for msg in state["messages"]]
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[system_prompt] + messages
        )
        classification = response['choices'][0]['message']['content'].strip()
        state["classification"] = classification
    
        if classification == "Direct_answer":
            system_prompt_direct = {
                "role": "system",
                "content": "You are a helpful assistant. Please provide a response to a user query in a friendly tone."
            }
            assistant_response = self.llm_service.generate_response(system_prompt_direct, state["messages"])
            state["assistant_response"] = assistant_response
        elif classification == "Unclear_intent":
            state["assistant_response"] = {
                "role": "assistant",
                "content": "I'm sorry, I couldn't understand your query. Could you clarify?",
            }
        elif classification == "Simple_question":
            user_query = state["messages"][-1].content
            context = self.tools.vector_search_tool(user_query)
            assistant_response = {
                "role": "assistant",
                "content": f"The retrieved context is:\n\n{context}",
            }
            state["assistant_response"] = assistant_response
        elif classification == "Complex_question":
            user_query = state["messages"][-1].content
            sub_questions = self.tools.deconstruct_question_tool(user_query)
            state["assistant_response"] = {
                "role": "assistant",
                "content": f"Complex question handling is a work in progress. Retrieved sub-questions:\n\n{sub_questions}",
            }
    
        return state


    def summarizer_agent(self, state: State) -> State:
        """
        Summarizes a document or multiple documents based on the user query.
    
        Args:
            state (State): The current state containing messages.
    
        Returns:
            State: Updated state containing the assistant's response.
        """
        LOGGER.debug(f"Entered summarizer_agent with state: {state}")
    
        # Extract file names from the state
        file_names = state.get("file_names", [])
    
        # Define an async function for summarization logic
        async def handle_summarizer_logic():
            summaries = await self.tools.summarize_tool(file_names)  # Ensure summarize_tool is async
            LOGGER.debug(f"Generated summaries: {summaries}")
            return summaries
    
        # Run the async function in an event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                summaries = asyncio.run_coroutine_threadsafe(handle_summarizer_logic(), loop).result(timeout=60)
            except asyncio.TimeoutError:
                LOGGER.error("Timeout while generating summaries.")
                summaries = {}
        else:
            summaries = loop.run_until_complete(handle_summarizer_logic())
    
        # Format the assistant response
        assistant_response = {
            "role": "assistant",
            "content": "The following summaries have been generated:\n\n - " +
                       "\n - ".join([f"{name}: {summary}\n\n" for name, summary in summaries.items()])
        }
    
        state["assistant_response"] = assistant_response
        LOGGER.debug(f"State after summarizer_agent processing: {state}")
        return state

    def multihop_agent(self, state: State) -> State:
        """
        Handles complex questions by breaking them down into simpler sub-questions.
        """
        LOGGER.debug("Within multihop_agent")
    
        # Deconstruct the complex question into simpler questions
        original_question = state["messages"][-1].content
        LOGGER.debug(f"Deconstructing the complex question: {original_question}")
        deconstructed_questions = self.tools.deconstruct_question_tool(original_question)
    
        # Ensure event loop exists
        ensure_event_loop()
    
        # Process each sub-question
        async def handle_multihop_logic():
            tasks = [self.answer_simpler_question_async(question) for question in deconstructed_questions]
            return await asyncio.gather(*tasks)
    
        loop = asyncio.get_event_loop()
        responses = loop.run_until_complete(handle_multihop_logic())
        LOGGER.debug(f"Responses for sub-questions: {responses}")
    
        # Combine the responses into a final answer
        final_answer = self.tools.generate_final_answer(original_question, responses)
        state["assistant_response"] = {"role": "assistant", "content": final_answer}
    
        return state


    async def answer_simpler_question_async(self, question: str) -> str:
        """
        Asynchronously handles a single simpler question.
    
        Args:
            question (str): The sub-question to answer.
    
        Returns:
            str: The generated answer for the sub-question.
        """
        try:
            LOGGER.debug(f"Starting vector search for question: {question}")
            context = await self.tools.vector_search_tool_async(question)  # Updated for async
            LOGGER.debug(f"Retrieved context: {context}")
    
            LOGGER.debug("Generating answer based on retrieved context.")
            answer = await self.tools.answer_generator_tool_async(question, context)
            LOGGER.debug(f"Generated answer: {answer}")
    
            return answer
        except Exception as e:
            LOGGER.error(f"Error processing question: {question}. Exception: {e}")
            return f"Error: Unable to process question '{question}'."
    


class Graph:
    def __init__(self, state_class):
        """
        Initializes the StateGraph with agents and their connections.
        """
        agents = Agents(OpenAI_LLM_Service())  # Initialize the Agents class
        conditions = Conditions()  # Initialize Conditions for decision-making logic

        # Initialize the StateGraph and configure nodes/edges
        graph_builder = StateGraph(state_class)

        # Add nodes for agents
        graph_builder.add_node("classifier_agent", agents.classifier_agent)
        graph_builder.add_node("summarizer_agent", agents.summarizer_agent)
        graph_builder.add_node("multihop_agent", agents.multihop_agent)

        # Define edges between nodes
        graph_builder.add_edge(START, "classifier_agent")
        graph_builder.add_conditional_edges("classifier_agent", conditions.decide_agent_based_on_class)
        graph_builder.add_edge("multihop_agent", END)
        graph_builder.add_edge("summarizer_agent", END)

        # Compile the graph with a MemorySaver for checkpoints
        self.compiled_graph = graph_builder.compile(checkpointer=MemorySaver())

        # Optionally display the graph structure (requires additional dependencies)
        try:
            display(Image(self.compiled_graph.get_graph().draw_mermaid_png()))
        except Exception:
            pass  # Handle cases where visualization tools are not installed

    def get_compiled_graph(self):
        """
        Returns the compiled StateGraph instance.
        """
        return self.compiled_graph

    async def stream_graph_updates(self, thread_id, user_input: str):
        LOGGER.debug(f"Starting stream_graph_updates with thread_id: {thread_id} and user_input: {user_input}")
        state = {"messages": [HumanMessage(content=user_input)]}
        config = {"configurable": {"thread_id": str(thread_id)}}
        try:
            result = await asyncio.to_thread(self.compiled_graph.invoke, state, config=config)
            LOGGER.debug(f"Graph execution result: {result}")
            classification = state.get("classification")
            LOGGER.info(f"Triggered classification: {classification}")
            assistant_response = result.get("assistant_response", {}).get("content")
            if assistant_response:
                print(f"\n\nAssistant: {assistant_response}")
        except Exception as e:
            LOGGER.error(f"Error during graph execution: {e}")






async def main():
    OpenAI_LLM = OpenAI_LLM_Service()
    agents = Agents(OpenAI_LLM)
    graph_instance = Graph(State)
    compiled_graph = graph_instance.get_compiled_graph()
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            thread_id = getpass.getuser()
            await graph_instance.stream_graph_updates(thread_id, user_input)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()  # Allows nested asyncio loops in environments like Jupyter
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except Exception as e:
        print(f"An error occurred: {e}")



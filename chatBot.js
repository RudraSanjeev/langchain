import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import readline from "readline";

// import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import { config } from "dotenv";
config();

// load data and create vector store
const createVectorStore = async () => {
  const loader = new CheerioWebBaseLoader(
    "https://js.langchain.com/docs/expression_language"
  );

  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 20,
  });

  const splitDocs = await splitter.splitDocuments(docs);

  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
  });

  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );

  return vectorStore; //db
};

// create retrival chain
const createChain = async () => {
  const model = new ChatGoogleGenerativeAI({
    modelName: "gemini-pro",
    temperature: 0.8, // set creativity 0 --> 1(most)
    maxOutputTokens: 300, // no of char
    apiKey: process.env.GEMINI_API_KEY,
  });

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the user's questions based on the following context: {context}.",
    ],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
  ]);

  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt,
  });

  const retriever = vectorStore.asRetriever({
    k: 2,
  });

  // const retrieverPrompt = ChatPromptTemplate.fromMessages([
  //   new MessagesPlaceholder("chat_history"),
  //   ["user", "{input}"],
  //   [
  //     "user",
  //     "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
  //   ],
  // ]);

  // const historyAwareRetriever = await createHistoryAwareRetriever({
  //   llm: model,
  //   retriever,
  //   rephrasePrompt: retrieverPrompt,
  // });

  const conversationChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever,
  });

  return conversationChain;
};

const vectorStore = await createVectorStore();
// console.log(vectorStore);

const chain = await createChain(vectorStore);

const chatHistory = [
  new HumanMessage("Hello"),
  new AIMessage("Hi, How can i help you ?"),
  new HumanMessage("My name is anna"),
  new AIMessage("Okay got it. your name is anna, How can i help you ?"),
  new HumanMessage("Let assume your name is muku"),
  new AIMessage("Okay !"),
  new HumanMessage("what is LCEL ?"),
  new AIMessage("LCEL stands for Lanfchain Expression Language"),
];
const res = await chain.invoke({
  input: "what can we do with LCEL?",
  chat_history: chatHistory,
});

console.log(res.answer);

import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";

import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";

// import { Document } from "@langchain/core/documents";

// Import environment variables
import { config } from "dotenv";
config();

// Instantiate Model

const model = new ChatGoogleGenerativeAI({
  modelName: "gemini-pro",
  temperature: 0.8,
  apiKey: process.env.GEMINI_API_KEY,
});
// Create prompt
const prompt = ChatPromptTemplate.fromTemplate(
  `Answer the user's question from the following context: 
  {context}
  Question: {input}`
);

// Create Chain
const chain = await createStuffDocumentsChain({
  llm: model,
  prompt,
});

// Manually create documents
// const documentA = new Document({
//   pageContent:
//     "LangChain Expression Language or LCEL is a declarative way to easily compose chains together. Any chain constructed this way will automatically have full sync, async, and streaming support. ",
// });

// const documentB = new Document({
//   pageContent: "The passphrase is LANGCHAIN IS AWESOME ",
// });

// Use Cheerio to scrape content from webpage and create documents
const loader = new CheerioWebBaseLoader(
  "https://js.langchain.com/docs/expression_language/"
);
const docs = await loader.load();

// Text Splitter
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 20,
});
const splitDocs = await splitter.splitDocuments(docs);
// console.log(splitDocs[splitDocs.length - 1].metadata.loc);

// Instantiate Embeddings function
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
});

// Create Vector Store
const vectorstore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);
// Create a retriever from vector store
const retriever = vectorstore.asRetriever({ k: 2 });
// console.log(retriever);
// Create a retrieval chain
const retrievalChain = await createRetrievalChain({
  combineDocsChain: chain,
  retriever,
});

// // // Invoke Chain
// // const response = await chain.invoke({
// //   question: "What is LCEL?",
// //   context: splitDocs,
// // });

console.log(retrievalChain);

// const response = await retrievalChain.invoke({
//   input: "What is LCEL?",
// });

// console.log(response.answer);

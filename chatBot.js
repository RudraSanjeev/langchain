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

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

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
      "Answer the user's questions only if found in the following context: {context} or in the chat_history: {chat_history}",
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
  // const v1 = retriever.lc_kwargs.vectorStore.memoryVectors[0].metadata.source;
  // const v2 = retriever.lc_kwargs.vectorStore.memoryVectors[1].metadata.source;
  // console.log(v1);
  // console.log(v2);
  // console.log(retriever.lc_kwargs.vectorStore.similarity(v1, v2));

  const retrieverPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the user's question: {input} based on the above conversation context.",
    ],
    new MessagesPlaceholder("chat_history"),
    [
      "user",
      "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation only if it present inside the context or chat_history",
    ],
  ]);

  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm: model,
    retriever,
    rephrasePrompt: retrieverPrompt,
  });

  const conversationChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever: historyAwareRetriever,
  });

  return conversationChain;
};

const vectorStore = await createVectorStore();
// console.log(vectorStore);

const chain = await createChain(vectorStore);

const chatHistory = [
  new HumanMessage("Hello"),
  new AIMessage("Hi, How can i help you ?"),
  new HumanMessage("My name is John Doe"),
  new AIMessage("Okay got it. your name is John Doe, How can i help you ?"),
  new HumanMessage("Let assume your name is muku"),
  new AIMessage("Okay !"),
  new HumanMessage("what is LCEL ?"),
  new AIMessage("LCEL stands for Lanfchain Expression Language"),
];

const restrictSentence =
  "Answer this question only if it is found inside the context or in the chat_history";

// const res = await chain.invoke({
//   // input: "who is the founder of google ?",
//   input: `what is huggingface ? ${restrictSentence}`,
//   // input: "what is your name?",
//   chat_history: chatHistory,
// });

// console.log(res.answer);
const chatBot = async (input) => {
  return new Promise(async (resolve) => {
    try {
      const res = await chain.invoke({
        input: input,
        chat_history: chatHistory,
      });
      // chatHistory.push(new AIMessage(res.answer));
      console.log(res.answer);
      resolve();
    } catch (error) {
      console.error(`Error in chatBot function: ${error}`);
      resolve(); // Resolve the promise even in case of an error
      // chatHistory.push(new AIMessage(res.answer));
    }
  });
};

const askQuestion = async () => {
  while (true) {
    try {
      const answer = await new Promise((resolve) => {
        rl.question("Enter a Question (or press Ctrl + C to exit): ", resolve);
      });

      await chatBot(answer);
      // chatHistory.push(new HumanMessage(answer));
    } catch (error) {
      console.error(`Error in askQuestion function: ${error}`);
      // chatHistory.push(new HumanMessage(answer));
    }
  }
};

askQuestion();

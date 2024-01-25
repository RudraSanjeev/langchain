// import { LLMChain } from "langchain";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder,
  SystemMessagePromptTemplate,
} from "@langchain/core/prompts";

import { config } from "dotenv";
config();
// LLM
// const llm = new LLMChain();

const llm = new ChatGoogleGenerativeAI({
  modelName: "gemini-pro",
  temperature: 0.3,
  apiKey: process.env.GEMINI_API_KEY,
});

// Prompt
const prompt = new ChatPromptTemplate([
  new SystemMessagePromptTemplate(
    "You are a nice chatbot having a conversation with a human."
  ),
  new MessagesPlaceholder("chat_history"),
  new HumanMessagePromptTemplate("{question}"),
]);

// Notice that we `return_messages=true` to fit into the MessagesPlaceholder
// Notice that `"chat_history"` aligns with the MessagesPlaceholder name
const memory = new ConversationBufferMemory("chat_history", true);
const conversation = new ChatGoogleGenerativeAI(llm, prompt, true, memory);

// Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
conversation({ question: "hi" });

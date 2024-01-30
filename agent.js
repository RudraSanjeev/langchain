import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { AgentExecutor, createOpenAIToolsAgent } from "langchain/agents";
import { ConversationChain } from "langchain/chains";
import { config } from "dotenv";
config();

const model = new ChatGoogleGenerativeAI({
  modelName: "gemini-pro",
  temperature: 0, // set creativity 0 --> 1(most)
  maxOutputTokens: 300, // no of char
  apiKey: process.env.GEMINI_API_KEY,
});

const prompt = ChatPromptTemplate.fromMessages([
  ("system", "You are helpful assistant called muku"),
  ("human", "{input}"),
  new MessagesPlaceholder("agent_scratchpad"),
  ("human", "thankyou for your response"),
]);

const travilySearch = new TavilySearchResults({
  apiKey: process.env.TRAVILY_SEARCH_API_KEY,
});
const tools = [travilySearch];

const agent = await createOpenAIToolsAgent({
  llm: model,
  prompt,
  tools,
});

const agentExecutor = new AgentExecutor({
  agent,
  tools,
});

console.log(agentExecutor);

const result = await agentExecutor.invoke({
  input: "what is the weather in california ?",
});

console.log(result);

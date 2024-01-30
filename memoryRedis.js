import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { UpstashRedisChatMessageHistory } from "@langchain/community/stores/message/upstash_redis";
import { config } from "dotenv";
config();
const memory = new BufferMemory({
  chatHistory: new UpstashRedisChatMessageHistory({
    sessionId: "123",
    config: {
      url: process.env.UPSTASH_REDIS_URI,
      token: process.env.UPSTASH_REDIS_TOKEN,
    },
  }),
});
const model = new ChatGoogleGenerativeAI({
  modelName: "gemini-pro",
  temperature: 0.8, // set creativity 0 --> 1(most)
  maxOutputTokens: 300, // no of char
  apiKey: process.env.GEMINI_API_KEY,
});

const chain = new ConversationChain({
  llm: model,
  memory,
});

const res = await chain.call({
  //   input: "Hi, my name is sanjeev",
  input: "How many types of it?",
});
console.log(res.response);

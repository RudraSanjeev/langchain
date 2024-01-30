import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { config } from "dotenv";

config();

const model = new ChatGoogleGenerativeAI({
  modelName: "gemini-pro",
  //temperature: 0.8, // set creativity 0 --> 1(most)
  maxOutputTokens: 25, // no of char
  apiKey: process.env.GEMINI_API_KEY,
  // streaming: true,
});

const response = await model.invoke("Tell me a joke.", {
  callbacks: [
    {
      handleLLMNewToken(token) {
        console.log({ token });
      },
    },
  ],
});
console.log(response);
// const response = await model.invoke("tell me a joke");
// console.log(response.content);

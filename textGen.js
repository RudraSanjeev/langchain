import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HumanMessage } from "@langchain/core/messages";
import { config } from "dotenv";
import readline from "readline";

config();

const llm = new ChatGoogleGenerativeAI({
  modelName: "gemini-pro",
  temperature: 0.3,
  apiKey: process.env.GEMINI_API_KEY,
});

const textGen = async (topic) => {
  const input = [
    new HumanMessage({
      content: [
        {
          type: "text",
          text: `generate large text docs around 1000 words on the topic ${topic}.`,
        },
      ],
    }),
  ];

  const res = await llm.invoke(input);

  console.log(res.content);
};

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

// Ask the user for input
rl.question("Enter a topic: ", async (topic) => {
  // Call textGen function with user input
  await textGen(topic);

  // Close the readline interface
  rl.close();
});

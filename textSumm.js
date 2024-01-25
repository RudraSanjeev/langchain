import { TextLoader } from "langchain/document_loaders/fs/text";
import { loadSummarizationChain } from "langchain/chains";
import { TokenTextSplitter } from "langchain/text_splitter";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { PromptTemplate } from "@langchain/core/prompts";
import { config } from "dotenv";
config();
const loader = new TextLoader("./demo.txt");

const docs = await loader.load();

const splitter = new TokenTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 100,
});

const docsSummary = await splitter.splitDocuments(docs);

const llmSummary = new ChatGoogleGenerativeAI({
  modelName: "gemini-pro",
  temperature: 0.3,
  apiKey: process.env.GEMINI_API_KEY,
});

const summaryTemplate = `
  You are an expert in summarizing text file.
  Your goal is to create a summary of a text of the file.
  Below you find the text of the file:
  --------
  {text}
  --------
  
  Total output will be a summary of the text document.
  
  SUMMARY:
  `;

const SUMMARY_PROMPT = PromptTemplate.fromTemplate(summaryTemplate);

const summaryRefineTemplate = `
  You are an expert in summarizing text document file.
  Your goal is to create a summary of a text document.
  We have provided an existing summary up to a certain point: {existing_answer}
  
  Below you find the text document of the file:
  --------
  {text}
  --------
  
  Given the new context, refine the summary.

  Total output will be a summary of the text document.
  
  SUMMARY:
  `;

const SUMMARY_REFINE_PROMPT = PromptTemplate.fromTemplate(
  summaryRefineTemplate
);

const summarizeChain = loadSummarizationChain(llmSummary, {
  type: "refine",
  //   verbose: true,
  questionPrompt: SUMMARY_PROMPT,
  refinePrompt: SUMMARY_REFINE_PROMPT,
});

const summary = await summarizeChain.run(docsSummary);
console.log(summary);

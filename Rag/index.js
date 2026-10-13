import { OllamaEmbeddings } from "@langchain/ollama";
import DocumentProcessor from "./src/DocumentProcessor.js";
import VectorDatabase from "./src/VectorDatabase.js";
import RagPipeline from "./src/RagPipeline.js";

async function run() {
  const connectionString = "postgresql://admin:admin@localhost:5432/ragdb";
  const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" });
  
  // 1. Initialize DB and Processor
  const db = new VectorDatabase(connectionString, embeddings);
  const processor = new DocumentProcessor("/Users/santhosh/Documents/genAIJS/Rag/docs");
  
  // 2. Process and Store (Run this once)
  console.log("Processing and indexing documents...");
  const docs = await processor.process();
  await db.index(docs);
  
  // 3. Query the Pipeline
  console.log("Executing RAG Query...");
  const vectorStore = await db.getStore();
  const rag = new RagPipeline(vectorStore, "qwen3:8b");
  
  const answer = await rag.ask("compare these 2 products Neon Samsung Galaxy S23 Plus Next and Nova Samsung Galaxy S23 Pro 2024");
  console.log("\nResponse:\n", answer);
}

run().catch(console.error);
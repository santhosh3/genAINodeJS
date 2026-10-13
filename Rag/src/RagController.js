import path from "path";
import { OllamaEmbeddings } from "@langchain/ollama";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";

import VectorDatabase from "./VectorDatabase.js";
import RagPipeline from "./RagPipeline.js";
import DocumentProcessor from "./DocumentProcessor.js";
import Database from "./Database.js";

const chatHistories = new Map();
const JSON_STORE_PATH = path.resolve("./docs/store.json");

class RagController {
  async ingestDocuments(_, res) {
    try {
      const folderPath = "/Users/santhosh/Documents/genAIJS/Rag/docs";
      console.log(`[API] Processing CSV & Documents from folder: "${folderPath}"`);

      const pool = Database.getPool();
      const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" });
      const db = new VectorDatabase(pool, embeddings);

      const processor = new DocumentProcessor(folderPath, JSON_STORE_PATH);
      const docs = await processor.process();

      if (!docs || docs.length === 0) {
        return res.status(400).json({ success: false, message: "No valid documents found to index." });
      }

      const totalIndexed = await db.indexDocuments(docs);

      return res.status(200).json({
        success: true,
        message: `Successfully generated JSON dataset and embedded ${totalIndexed} document chunks into pgvector.`,
        chunksIndexed: totalIndexed,
        jsonPath: JSON_STORE_PATH
      });
    } catch (error) {
      console.error("[API] Error in ingestDocuments:", error);
      return res.status(500).json({ success: false, error: error.message });
    }
  }

  async askQuestionStream(req, res) {
    try {
      const { question, sessionId = "default-session" } = req.body;

      if (!question) {
        return res.status(400).json({ error: "The 'question' field is required." });
      }

      console.log(`[API] Processing stream for session "${sessionId}": "${question}"`);

      const pool = Database.getPool();
      const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" });
      const db = new VectorDatabase(pool, embeddings);
      const vectorStore = await db.getStore();
      
      const rag = new RagPipeline(vectorStore, "qwen3:8b", JSON_STORE_PATH);

      const chainWithHistory = new RunnableWithMessageHistory({
        runnable: rag.chain,
        getMessageHistory: (id) => {
          if (!chatHistories.has(id)) {
            chatHistories.set(id, new InMemoryChatMessageHistory());
          }
          return chatHistories.get(id);
        },
        inputMessagesKey: "question",
        historyMessagesKey: "chat_history",
      });

      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      });

      const stream = await chainWithHistory.stream(
        { question },
        { configurable: { sessionId } }
      );

      for await (const chunk of stream) {
        const content = typeof chunk === "string" ? chunk : chunk?.content || "";
        if (content) {
          res.write(`data: ${JSON.stringify({ chunk: content })}\n\n`);
        }
      }

      res.write("data: [DONE]\n\n");
      res.end();

    } catch (error) {
      console.error("[API] Error in askQuestionStream:", error);
      if (!res.headersSent) {
        return res.status(500).json({ error: "Internal server error" });
      }
      res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
      res.end();
    }
  }
}

export default new RagController();
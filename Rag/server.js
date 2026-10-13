import express from "express";
import cors from "cors";
import RagController from "./src/RagController.js";
import Database from "./src/Database.js";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Routes
app.post("/api/ingest", (req, res) => RagController.ingestDocuments(req, res));
app.post("/api/ask", (req, res) => RagController.askQuestionStream(req, res));

// Graceful Shutdown
process.on("SIGINT", async () => {
  console.log("\nShutting down server...");
  const pool = Database.getPool();
  await pool.end();
  console.log("Database pool closed.");
  process.exit(0);
});

app.listen(PORT, () => {
  console.log(`🚀 RAG API Server running on http://localhost:${PORT}`);
  console.log(`  - Ingest API: POST http://localhost:${PORT}/api/ingest`);
  console.log(`  - Query SSE API: POST http://localhost:${PORT}/api/ask`);
});
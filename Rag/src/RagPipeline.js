import { ChatOllama } from "@langchain/ollama";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableBranch } from "@langchain/core/runnables";
import alasql from "alasql";
import fs from "node:fs/promises";
import path from "node:path";

class RagPipeline {
    constructor(vectorStore, llmModel = "qwen3:8b", jsonFilePath = "./docs/store.json") {
        this.jsonFilePath = path.resolve(jsonFilePath);

        this.retriever = vectorStore.asRetriever({
            searchType: "mmr",
            k: 4,
            fetchK: 20,
            lambda: 0.5,
        });

        this.llm = new ChatOllama({ model: llmModel, temperature: 0 });
        this.chain = this._buildChain();
    }

    _buildChain() {
        // Step 1: Query Router Classifier (Updated with Distinct / Listing criteria)
        const routerPrompt = ChatPromptTemplate.fromMessages([
            [
                "system",
                `Analyze the user question and classify it into one category:
                - Reply "AGGREGATION" if the question asks for structured dataset queries, distinct lists (e.g., "list categories", "show brands"), counts, totals, sums, averages, min/max, or structured filtering.
                - Reply "VECTOR" ONLY for text search, semantic explanations, specific product details, or qualitative context retrieval.

                Output ONLY 'AGGREGATION' or 'VECTOR'.`
            ],
            ["human", "{question}"]
        ]);

        const classifierChain = routerPrompt.pipe(this.llm).pipe(new StringOutputParser());

        // Step 2A: Vector RAG Branch
        const vectorPrompt = ChatPromptTemplate.fromMessages([
            [
                "system",
                `You are a helpful assistant. Answer the user's question based ONLY on the following context:\n\n{context}`
            ],
            new MessagesPlaceholder("chat_history"),
            ["human", "{question}"]
        ]);

        const vectorBranch = RunnablePassthrough.assign({
            context: async (input) => {
                const query = typeof input === "string" ? input : input.question;
                const docs = await this.retriever.invoke(query);
                const contextStr = docs
                    .map((d) => `[Source: ${d.metadata?.source || "N/A"}]\n${d.pageContent}`)
                    .join("\n\n---\n\n");

                console.log("\n================ [LOG] RETRIEVED VECTOR CONTEXT ================");
                console.log("Question:", query);
                console.log("Context:\n", contextStr);
                console.log("=================================================================\n");

                return contextStr;
            }
        })
            .pipe(vectorPrompt)
            .pipe(this.llm)
            .pipe(new StringOutputParser());

        const sqlGenPrompt = ChatPromptTemplate.fromMessages([
            [
                "system",
                `You are a SQL generator for an in-memory JSON array using AlaSQL syntax. Use '?' as the table name.

                Dataset Schema (available keys in '?'):
                - id (string)
                - name (string)
                - description (string)
                - price (string)
                - image_url (string)
                - category (string) (e.g., "Laptops", "Smartphones")

                CRITICAL RULES FOR STRING FILTERING:
                1. NEVER use exact equality (=) for string filters (e.g. DO NOT use category = 'laptop').
                2. ALWAYS use case-insensitive wildcard search using LOWER() and LIKE '%...%'.
                3. Account for singular/plural variations (e.g., searching for "laptop" should match "Laptops").
                4. Select concise columns (e.g., name, price, category) rather than returning giant image URLs unless asked.

                Query Examples:
                - "list all laptops" -> SELECT name, category, price FROM ? WHERE LOWER(category) LIKE '%laptop%' OR LOWER(name) LIKE '%laptop%'
                - "list all categories" -> SELECT DISTINCT category FROM ?
                - "count smartphones" -> SELECT COUNT(*) AS total FROM ? WHERE LOWER(category) LIKE '%smartphone%'
                - "cheapest laptop" -> SELECT name, price FROM ? WHERE LOWER(category) LIKE '%laptop%' ORDER BY CAST(price AS NUMBER) ASC LIMIT 1

                Output ONLY the raw AlaSQL query string without markdown formatting or extra explanations.`
            ],
            ["human", "{question}"]
        ]);

        const aggregationBranch = RunnablePassthrough.assign({
            sqlResult: async (input) => {
                const query = typeof input === "string" ? input : input.question;
                const rawSql = await sqlGenPrompt.pipe(this.llm).pipe(new StringOutputParser()).invoke({ question: query });
                const cleanSql = rawSql.replace(/```sql|```/g, "").trim();

                console.log("\n================ [LOG] EXECUTING ALASQL QUERY ================");
                console.log("Generated SQL:", cleanSql);

                try {
                    const rawData = await fs.readFile(this.jsonFilePath, "utf-8");
                    const dataset = JSON.parse(rawData);
                    const result = alasql(cleanSql, [dataset]);
                    const resultStr = JSON.stringify(result);

                    console.log("AlaSQL Result:", resultStr);
                    console.log("===============================================================\n");
                    return resultStr;
                } catch (err) {
                    console.error("AlaSQL Error:", err.message);
                    return `Error executing AlaSQL query: ${err.message}`;
                }
            }
        })
            .pipe(
                ChatPromptTemplate.fromMessages([
                    [
                        "system",
                        `Answer the user's question clearly based on this JSON query result:\n\n{sqlResult}`
                    ],
                    new MessagesPlaceholder("chat_history"),
                    ["human", "{question}"]
                ])
            )
            .pipe(this.llm)
            .pipe(new StringOutputParser());

        // Step 3: Combine via RunnableBranch
        return RunnableBranch.from([
            [
                async (input) => {
                    const classification = await classifierChain.invoke(input);
                    const isAggregation = classification.trim().toUpperCase().includes("AGGREGATION");
                    console.log(`[LOG] Router Selected: ${isAggregation ? "AGGREGATION ENGINE" : "VECTOR RAG"}`);
                    return isAggregation;
                },
                aggregationBranch
            ],
            vectorBranch
        ]);
    }

    async ask(question) {
        return await this.chain.invoke({ question });
    }
}

export default RagPipeline;
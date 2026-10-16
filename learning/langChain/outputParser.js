import { z } from "zod";
import { initChatModel } from "langchain/chat_models/universal";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// 1. Define your Zod schema
// Adding .describe() gives the model helpful context about each field
const MovieAnalysisSchema = z.object({
    title: z.string().describe("The official title of the movie"),
    releaseYear: z.number().describe("The year the movie was released"),
    genres: z.array(z.string()).describe("List of main genres"),
    summary: z.string().describe("A brief 2-sentence plot summary"),
    rating: z.number().min(1).max(10).describe("Rating from 1 to 10"),
});

async function runStructuredExample() {
    // 2. Initialize the model
    const model = await initChatModel("qwen3:1.7b", {
        modelProvider: "ollama",
        temperature: 0,
    });

    // 3. Bind the schema to the model
    const structuredModel = model.withStructuredOutput(MovieAnalysisSchema);

    // 4. Invoke the structured model
    const response = await structuredModel.invoke(
        "Analyze the movie Inception directed by Christopher Nolan."
    );

    console.log("Structured JSON Output:\n", response);
    console.log("\nType checking test - Movie Title:", response.title);
}



async function runChainExample() {
    // Schema for extracting person details
    const PersonExtractorSchema = z.object({
        fullName: z.string().describe("Full name of the person"),
        age: z.number().optional().describe("Age if mentioned"),
        skills: z.array(z.string()).describe("Skills or technologies mentioned"),
        location: z.string().nullable().describe("Current city/country if mentioned"),
    });

    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "Extract all personal details from the given text."],
        ["user", "{bio}"],
    ]);

    const model = await initChatModel("qwen3:1.7b", {
        modelProvider: "ollama",
        temperature: 0,
    });
    
    // Create structured model wrapper
    const structuredModel = model.withStructuredOutput(PersonExtractorSchema);

    // Pipe prompt -> structured model
    const chain = prompt.pipe(structuredModel);

    const result = await chain.invoke({
        bio: "Santhosh is a 26-year-old software engineer based in Bengaluru. He specializes in Node.js, TypeScript, and LangChain.",
    });

    console.log("Extracted Data:", result);
}

runChainExample().catch(console.error);
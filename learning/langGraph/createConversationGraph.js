import { StringOutputParser } from "@langchain/core/output_parsers";
import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import { HumanMessage, SystemMessage, AIMessage } from "langchain";
import { ChatOllama } from "@langchain/ollama";
import { writeFile } from "node:fs/promises";
import path from "node:path";

const MODEL = "qwen3:8b";
const model = new ChatOllama({ model: MODEL, temperature: 0 });
const ConversationState = Annotation.Root({
    messages: Annotation({
        reducer: (current, update) => current.concat(update),
        default: () => [],
    }),

    sentiment: Annotation({
        reducer: (_, update) => update,
        default: () => "",
    }),

    count: Annotation({
        reducer: (_, update) => update,
        default: () => 0,
    }),
});

class ConversationGraph {

    async analyzeSentiment(state) {
        const last_message = state.messages.at(-1);
        const chain = model.pipe(new StringOutputParser());
        const response = await chain.invoke([
            new SystemMessage(
                "Classify sentiment as positive, negative, or neutral. " +
                "Reply with just the word."
            ),
            new HumanMessage(last_message.content),
        ]);
        let sentiment = response.trim().toLowerCase();
        if (!["positive", "negative", "neutral"].includes(sentiment)) {
            sentiment = "neutral";
        }
        return { sentiment }
    }

    async generateResponse(state) {
        const sentiment = state.sentiment;
        const lastMessage = state.messages.at(-1);
        const systemPrompts = {
            positive: "Respond enthusiastically and build on the user's positive energy.",
            negative: "Respond empathetically and offer support.",
            neutral: "Respond helpfully and informatively.",
        }
        const prompt = systemPrompts[sentiment] ?? systemPrompts.neutral;
        const response = await model.invoke([
            new SystemMessage(prompt),
            new HumanMessage(lastMessage.content),
        ]);
        return {
            messages: [
                new AIMessage(response.content),
            ],
            count: state.count + 1,
        };
    }

    async createGraph() {
        const graph = new StateGraph(ConversationState);
        graph.addNode(
            "analyze_sentiment",
            this.analyzeSentiment.bind(this)
        )
        graph.addNode(
            "generate_response",
            this.generateResponse.bind(this)
        )
        graph.addEdge(START, "analyze_sentiment")
        graph.addEdge("analyze_sentiment", "generate_response")
        graph.addEdge("generate_response", END)

        const app = graph.compile();
        const drawableGraph = await app.getGraphAsync();
        const image = await drawableGraph.drawMermaidPng();
        const filePath = path.resolve(
            process.cwd(),
            "conversation_graph.png"
        )
        await writeFile(
            filePath,
            Buffer.from(await image.arrayBuffer())
        )
        return app;
    }
}

async function demoConversation() {
    const conversation = new ConversationGraph();
    const app = await conversation.createGraph();
    const testMessage = [
        "I just got promoted at work! I'm so excited!",
        "My computer crashed and I lost all my work...",
        "What's the weather like today?",
    ];

    for (let msg of testMessage) {
        const result = await app.invoke({
            messages: [new HumanMessage(msg)],
            sentiment: "",
            count: 0,
        })
        console.log("Input:", msg);
        console.log("Sentiment:", result.sentiment);
        console.log(
            "Response:",
            result.messages[result.messages.length - 1].content
        );
        console.log("Count:", result.count);
        console.log("-".repeat(40));
    }
}

demoConversation().catch(console.error);
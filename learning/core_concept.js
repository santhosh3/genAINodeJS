import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser, JsonOutputParser } from "@langchain/core/output_parsers";
// const { zodToJsonSchema } = require("zod-to-json-schema");
import { initChatModel } from "langchain";
import {
    AIMessage,
    ToolMessage,
    SystemMessage,
    HumanMessage,
    ChatMessage,
} from "@langchain/core/messages";


async function demoBasichain() {
    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "You are a helpful assistant. Answer in one sentence"],
        ["human", "{question}"],
    ]);

    const model = new ChatOllama({
        model: "qwen3:1.7b",
        temperature: 0.7,
    });

    const parser = new StringOutputParser();

    const chain = prompt.pipe(model).pipe(parser);

    const result = await chain.invoke({
        question: "What is LangChain?",
    });

    return result

}

async function demoBatchchain() {
    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "You are a helpful assistant. Answer in one sentence."],
        ["human", "{question}"],
    ]);

    const model = new ChatOllama({
        model: "qwen3:1.7b",
        temperature: 0.7,
    });

    const inputs = [
        { "question": "Hello, How are you?" },
        { "question": "What is Your name?" },
        { "question": "What are you good at?" }
    ];

    const parser = new StringOutputParser();

    const chain = prompt.pipe(model).pipe(parser);

    const result = chain.batch(inputs)

    return result
}

async function demoStreamExecution() {
    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "Write a haiku about nature."],
        ["user", "{question}"]
    ]);

    const model = new ChatOllama({
        model: "qwen3:1.7b",
        temperature: 0.7,
    });

    const parser = new StringOutputParser();

    const chain = prompt.pipe(model).pipe(parser);

    for await (const chunk of await chain.stream({ question: "nature" })) {
        process.stdout.write(chunk);
    }
}

async function runFlexibleChain() {
    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "You are a helpful assistant. Answer in one sentence. Return your answer ONLY as valid JSON."],
        ["user", "{question}"]
    ]);

    const model = await initChatModel("qwen3:1.7b", {
        modelProvider: "ollama",
        temperature: 0.7,
    });

    const parser = new JsonOutputParser();

    const chain = prompt.pipe(model).pipe(parser);

    const inputs = [
        {"question": "What is LangChain?"},
        {"question": "How does dynamic model routing work? explain clearly with examples"}
    ]

    const results = chain.batch(inputs)

    return results

}

async function messages() {
    const messages = [
          new SystemMessage("You are a helpful assistant."),
          new HumanMessage("What is LangChain?"),
          new AIMessage("LangChain is a framework for building LLM applications."),
          new HumanMessage("What can I build with it?")
    ];
    const model = await initChatModel("qwen3:1.7b", {
        modelProvider: "ollama",
        temperature: 0.7,
    });
    const response = await model.invoke(messages);
    console.log(response.content);
}


async function runChatChain() {
    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "You are a {role}."],
        ["human", "{question}"]
    ]);
    const model = new ChatOllama({
        model: "qwen3:1.7b",
        temperature: 0.7
    });
    const parser = new StringOutputParser();
    const chain = prompt
        .pipe(model)
        .pipe(parser);

    const result = await chain.invoke({role: "helpful AI teacher", question: "What is artificial intelligence?"});
    return result;
}

runChatChain().then(res => console.log(res));






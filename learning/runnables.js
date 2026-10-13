import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableBranch } from "@langchain/core/runnables";
import { ChatOllama } from "@langchain/ollama";
import { initChatModel } from "langchain/chat_models/universal";


async function parallelRunnable() {
    const model = new ChatOllama({
        model: "qwen3:1.7b",
        temperature: 0.7,
    });

    const parser = new StringOutputParser();
    const summaryChain = ChatPromptTemplate
        .fromTemplate(
            "Summarize this in one sentence:\n{text}"
        )
        .pipe(model)
        .pipe(parser);

    const keywordChain = ChatPromptTemplate
        .fromTemplate(
            "Give me 3 keywords for this:\n{text}"
        )
        .pipe(model)
        .pipe(parser);

    const parallelChain = new RunnableParallel({
        steps: {
            summary: summaryChain,
            keyword: keywordChain
        }
    });

    const result = await parallelChain.invoke({
        text: "LangChain is a framework for developing applications powered by language models."
    });

    return result;
}


async function runnableLambda() {
    const cleanText = (text) => text.trim().toLowerCase();
    const model = new ChatOllama({
        model: "qwen3:1.7b",
        temperature: 0.7,
    });
    const parser = new StringOutputParser();
    const chain = new RunnableLambda({
        func: cleanText
    })
        .pipe(model)
        .pipe(parser);
    const result = await chain.invoke("   What is LangChain?   ");
    return result;
}

async function runnablePassThrough() {
    const chain = new RunnableParallel({
        steps: {
            original: new RunnablePassthrough(),
            upperCase: new RunnableLambda({
                func: (x) => x.toUpperCase()
            }),
            length: new RunnableLambda({
                func: (x) => x.length
            }),
        }
    })
    return await chain.invoke("Data")
}

async function demoPassThroughChain() {
    const prompt = ChatPromptTemplate
        .fromTemplate(
            "Original question: {question}\n Context: {context}\n\n Answer the question based on the context."
        )

    const chain = new RunnableParallel({
        steps: {
            context: new RunnableLambda({
                func: () => "LangChain was created by Harrison Chase in 2022."
            }),
            question: new RunnablePassthrough()
        }
    })
        .pipe(new RunnableLambda({
            func: (x) => ({
                context: x['context'],
                question: x['question']['question']
            })
        }))
        .pipe(prompt)
        .pipe(new ChatOllama({ model: "qwen3:1.7b", temperature: 0.7 }))
        .pipe(new StringOutputParser());

    const result = chain.invoke({ "question": "when LangChain is created just year?" });
    return result;
}

async function demoChainBranching() {
    // Note: Ensure your local Ollama has this exact model name or use a standard tag like "qwen2.5:1.5b"
    const model = new ChatOllama({ model: "qwen3:1.7b" });

    const codePrompt = ChatPromptTemplate.fromTemplate(
        "You are a coding expert. Help with: {input}"
    );

    const generalPrompt = ChatPromptTemplate.fromTemplate(
        "You are a helpful assistant. Answer: {input}"
    );

    // FIX 1: Corrected snake_case from_template to camelCase fromTemplate
    const classifier_prompt = ChatPromptTemplate.fromTemplate(
        "Classify this as 'code' or 'general': {input}\nReturn only the classification."
    );

    const classifier = classifier_prompt
        .pipe(model)
        .pipe(new StringOutputParser());

    // FIX 2: Modern routing using RunnableLambda instead of RunnableBranch.
    // RunnableBranch conditions CANNOT await an async invoke call natively.
    const router = RunnableLambda.from(async (data) => {
        // Await the classifier model's response
        const classification = await classifier.invoke(data);

        // Clean up the model output and check classification
        if (classification.toLowerCase().includes("code")) {
            return codePrompt.pipe(model).pipe(new StringOutputParser());
        } else {
            return generalPrompt.pipe(model).pipe(new StringOutputParser());
        }
    });

    const questions = [
        "How do I write a for loop in Python?",
        "What's the weather like today?",
    ];

    // FIX 3: Changed 'in' to 'of' to get the actual strings instead of array indexes (0, 1)
    // FIX 4: Added 'await' inside the loop because branch/router execution is asynchronous
    for (let q of questions) {
        console.log(`--- Processing Question: "${q}" ---`);
        const result = await router.invoke({ input: q });
        console.log(result);
    }
}


async function runAdvancedChain() {
    async function mockRetriever(query) {
        return "Node.js is an open-source, cross-platform JavaScript runtime environment.";
    }
    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "Answer the question based only on the context below:\nContext: {context}"],
        ["user", "Question: {question}"],
    ]);

    const model = await initChatModel("qwen3:1.7b", { modelProvider: "ollama" });

    // Chain that dynamically maps inputs
    const chain = RunnablePassthrough.assign({
        context: async (input) => mockRetriever(input.question),
    })
        .pipe(prompt)
        .pipe(model)
        .pipe(new StringOutputParser());

    const result = await chain.invoke({ question: "What is Node.js?" });
    console.log("Answer:", result);
}

runAdvancedChain();

import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import { ChatOllama } from "@langchain/ollama";


const MODEL = "qwen3:8b";
const model = new ChatOllama({ model: MODEL, temperature: 0 });

const RouterState = Annotation.Root({
    query: Annotation({
        reducer: (_, update) => update,
        default: () => "",
    }),

    query_type: Annotation({
        reducer: (_, update) => update,
        default: () => "",
    }),

    response: Annotation({
        reducer: (_, update) => update,
        default: () => "",
    }),
});


class demoBasicResponse {

    // Node functions
    async classify_query(state) {
        const response = await model.invoke(`
            Classify the following query as exactly one of:
            question, command, or statement.
            Reply with only the category word.
            Query: ${state.query}
    `);

        const queryType = response.content
            .toLowerCase()
            .trim()
            .replace(/[^a-z]/g, "");

        return {
            query_type: ["question", "command", "statement"].includes(queryType)
                ? queryType
                : "statement",
        };
    }

    async handle_question(state) {
        const response = await model.invoke(`Answer the question: ${state['query']}`);
        return { "response": `[Answer] ${response.content}` }
    }

    handle_command(state) {
        return { "response": `[Executing] I'll help you with: ${state['query']}` }
    }

    handle_statement(state) {
        return { "response": `[Acknowledged] Thanks for sharing: ${state['query']}` }
    }

    async createGraph() {

        // initalize graphs
        const graph = new StateGraph(RouterState);

        // attach nodes
        graph.addNode("classify_query", this.classify_query.bind(this))
        graph.addNode("handle_question", this.handle_question.bind(this))
        graph.addNode("handle_command", this.handle_command.bind(this))
        graph.addNode("handle_statement", this.handle_statement.bind(this))

        // create edges
        graph.addEdge(START, "classify_query")
        graph.addConditionalEdges(
            "classify_query",
            (state) => {
                const qt = state.query_type;
                if (qt === "question") return "question";
                if (qt === "command") return "command";
                return "statement"; // Default route
            },
            {
                question: "handle_question",
                command: "handle_command",
                statement: "handle_statement",
            }
        );
        graph.addEdge("handle_question", END)
        graph.addEdge("handle_command", END)
        graph.addEdge("handle_statement", END)

        // compile
        return graph.compile();
    }
}


async function demoRoutingCondEdge() {
    const conversation = new demoBasicResponse();
    const app = await conversation.createGraph();

    const queries = [
        "What is the capital of France?",
        "Send an email to John",
        "I love programming",
    ];

    for (const query of queries) {
        const result = await app.invoke({ query });

        console.log("Input:", query);
        console.log("Type:", result.query_type);
        console.log("Response:", result.response);
        console.log("-".repeat(40));
    }
}


const qualityState = Annotation.Root({
    content: Annotation({
        reducer: (_, update) => update,
        default: () => "",
    }),

    feedback: Annotation({
        reducer: (_, update) => update,
        default: () => "",
    }),

    final_content: Annotation({
        reducer: (_, update) => update,
        default: () => "",
    }),

    quality_score: Annotation({
        reducer: (_, update) => update,
        default: () => 0,
    }),

    iteration: Annotation({
        reducer: (_, update) => update,
        default: () => 0,
    }),
});

class DemoImproveContentScore {

    // Node 1: Evaluate content quality
    async evaluate_quality(state) {
        const response = await model.invoke(`
            Rate this content quality from 1 to 10.
            Reply with only a single number.

            Content: ${state.content}
        `);

        // Extract and validate the score
        const match = String(response.content)
            .trim()
            .match(/\b(10|[1-9])\b/);

        const score = match ? Number(match[0]) : 5;

        console.log(`Quality score: ${score}/10`);

        return {
            quality_score: score,
        };
    }

    // Node 2: Improve content
    async improve_content(state) {
        console.log(
            `Improving content - iteration ${state.iteration + 1}`
        );

        const response = await model.invoke(`
            Improve this content to make it more engaging and clear.
            Preserve its original meaning.
            Return only the improved content.

            Content: ${state.content}
        `);

        return {
            content: String(response.content).trim(),
            iteration: state.iteration + 1,
        };
    }

    // Node 3: Finalize content
    finalize_content(state) {
        const reachedTarget = state.quality_score >= 7;

        return {
            final_content: state.content,
            feedback: reachedTarget
                ? `Target quality reached after ${state.iteration} improvement(s).`
                : `Maximum iterations reached. Final score: ${state.quality_score}/10.`,
        };
    }

    createGraph() {
        const graph = new StateGraph(qualityState);

        // Register nodes
        graph.addNode(
            "evaluate",
            this.evaluate_quality.bind(this)
        );

        graph.addNode(
            "improve",
            this.improve_content.bind(this)
        );

        graph.addNode(
            "finalize",
            this.finalize_content.bind(this)
        );

        // Define edges
        graph.addEdge(START, "evaluate");

        // Conditional routing
        graph.addConditionalEdges(
            "evaluate",
            (state) => {
                if (state.quality_score >= 7) {
                    return "finalize";
                }

                if (state.iteration >= 3) {
                    return "finalize";
                }

                return "improve";
            },
            {
                improve: "improve",
                finalize: "finalize",
            }
        );

        // Loop back to evaluation after improving
        graph.addEdge("improve", "evaluate");

        graph.addEdge("finalize", END);

        return graph.compile();
    }
}

async function demoImproveContent() {
    const conversation = new DemoImproveContentScore();
    const app = conversation.createGraph();

    const originalContent = "`The product is good and stuff happens and people use it and things are nice.`";

    const result = await app.invoke({
        content: originalContent,
        quality_score: 0,
        feedback: "",
        final_content: "",
        iteration: 0,
    });

    console.log("\n========== FINAL RESULT ==========");
    console.log("Original:", originalContent);
    console.log("Final:", result.final_content);
    console.log("Score:", `${result.quality_score}/10`);
    console.log("Iterations:", result.iteration);
    console.log("Feedback:", result.feedback);
}

demoImproveContent().catch(console.error);



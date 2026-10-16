import { StateGraph, START, END } from "@langchain/langgraph";
import * as z from "zod";

const simpleState = z.object({
    input: z.string(),
    output: z.string(),
    step: z.number(),
})

class SimpleGraph {
    // create node function
    process(state) {
        return {
            output: state.input.toUpperCase(),
            step: state.step + 1
        }
    }

    // create graph
    createGraph() {
        const graph = new StateGraph(simpleState);

        // Add node
        graph.addNode("process", this.process.bind(this));

        // Add edge
        graph.addEdge(START, "process");
        graph.addEdge("process", END);

        // compile
        return graph.compile();
    }

    async demo() {
        const app = this.createGraph();

        const result = await app.invoke({
            input: "hello",
            output: "",
            step: 0,
        })

        console.log("Simple graph result:", result);
        console.log(
            `Input: ${result.input}, Output: ${result.output}, Step: ${result.step}`
        );
    }
}

// Run the example
const simpleGraph = new SimpleGraph();
await simpleGraph.demo();
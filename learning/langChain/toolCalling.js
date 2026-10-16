import { tool } from "@langchain/core/tools";
import { HumanMessage, ToolMessage, SystemMessage } from "@langchain/core/messages";
import { initChatModel } from "langchain/chat_models/universal";
import { z } from "zod";

// --- 1. Tools with Strict Schemas & Coercion ---

const searchUserTool = tool(
  async ({ name }) => {
    const users = {
      Santhosh: { userId: "USR_101", city: "Bengaluru", membership: "Gold" },
      Rahul: { userId: "USR_102", city: "Mumbai", membership: "Silver" },
    };
    const user = users[name];
    if (!user) return `User "${name}" not found.`;
    return JSON.stringify(user);
  },
  {
    name: "search_user",
    description: "Finds a user by name and returns their user ID, city, and membership status.",
    schema: z.object({
      name: z.string().describe("Exact full name of the user to search for."),
    }),
  }
);

const getUserOrdersTool = tool(
  async ({ userId }) => {
    const orders = {
      USR_101: [{ orderId: "ORD_991", amount: 2500, status: "Delivered" }],
      USR_102: [{ orderId: "ORD_992", amount: 800, status: "Pending" }],
    };
    const userOrders = orders[userId];
    if (!userOrders) return `No orders found for user ID: ${userId}`;
    return JSON.stringify(userOrders);
  },
  {
    name: "get_user_orders",
    description: "Fetches recent purchase order history using a user ID.",
    schema: z.object({
      userId: z
        .string()
        .describe("The exact user ID returned by search_user (e.g. USR_101). DO NOT guess or use placeholders."),
    }),
  }
);

const calculateDiscountTool = tool(
  async ({ amount, membership }) => {
    const discountPercent = membership.toLowerCase() === "gold" ? 20 : 5;
    const finalAmount = amount - (amount * discountPercent) / 100;
    return `Original: ₹${amount}, Membership: ${membership}, Discount: ${discountPercent}%, Final Price: ₹${finalAmount}`;
  },
  {
    name: "calculate_discount",
    description: "Calculates discounted price based on total order amount and membership level.",
    schema: z.object({
      amount: z.coerce.number().describe("Numerical order amount in INR."),
      membership: z.string().describe("Membership status: Gold or Silver."),
    }),
  }
);

// --- 2. Main Agent Execution Loop ---

async function runMultiToolAgentLoop() {
  // Initialize Local Ollama Model
  const model = await initChatModel("qwen3:8b", {
    modelProvider: "ollama",
    temperature: 0,
  });

  const toolsList = [searchUserTool, getUserOrdersTool, calculateDiscountTool];
  const toolMap = Object.fromEntries(toolsList.map((t) => [t.name, t]));

  // Bind tools
  const modelWithTools = model.bindTools(toolsList, { parallel_tool_calls: false });

  // Add SystemMessage to guide step-by-step execution (vital for 1B-3B models)
  const messages = [
    new SystemMessage(
      "You are a helpful assistant with access to tools.\n" +
        "Rules:\n" +
        "1. Execute tool calls sequentially ONE BY ONE.\n" +
        "2. Do NOT guess parameters (like user IDs). Always retrieve them first via tools.\n" +
        "3. If a tool does not exist, acknowledge the failure and proceed with the remaining request."
    ),
    new HumanMessage(
      "Please find user Santhosh's user ID, fetch his order history, calculate his discounted price for that order."
    ),
  ];

  const maxIterations = 10;
  let iteration = 0;

  console.log("=== Multi-Tool Agent Execution Started ===\n");

  while (iteration < maxIterations) {
    iteration++;
    console.log(`--- Iteration ${iteration} ---`);

    const aiResponse = await modelWithTools.invoke(messages);
    messages.push(aiResponse);

    const toolCalls = aiResponse.tool_calls;

    // Exit condition: No further tool calls
    if (!toolCalls || toolCalls.length === 0) {
      console.log("\n================ Final AI Response ================");
      console.log(aiResponse.content);
      break;
    }

    console.log(`Model requested ${toolCalls.length} tool call(s):`);

    for (const toolCall of toolCalls) {
      console.log(` -> Tool: "${toolCall.name}" | ID: ${toolCall.id}`);
      console.log(`    Args: ${JSON.stringify(toolCall.args)}`);

      const targetTool = toolMap[toolCall.name];
      let toolOutputContent = "";

      if (!targetTool) {
        console.warn(`    ⚠️ [Hallucination Detected] Tool "${toolCall.name}" does NOT exist.`);
        toolOutputContent = `Error: Tool "${toolCall.name}" does not exist. Available tools: ${Object.keys(toolMap).join(", ")}. Please skip this step and continue.`;
      } else {
        try {
          toolOutputContent = await targetTool.invoke(toolCall.args);
          console.log(`    ✅ Result: ${toolOutputContent}`);
        } catch (err) {
          toolOutputContent = `Error executing tool: ${err.message}`;
          console.error(`    ❌ Execution Error: ${err.message}`);
        }
      }

      // Append ToolMessage to maintain valid message history chain
      messages.push(
        new ToolMessage({
          content: String(toolOutputContent),
          tool_call_id: toolCall.id,
        })
      );
    }
    console.log("\n");
  }

  if (iteration >= maxIterations) {
    console.log("Reached maximum iteration safety cap.");
  }
}

runMultiToolAgentLoop().catch(console.error);
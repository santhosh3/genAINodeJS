import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { initChatModel } from "langchain/chat_models/universal";

// 1. Create a store to manage session histories (In-Memory Map)
const messageHistories = {};

function getMessageHistory(sessionId) {
  if (!(sessionId in messageHistories)) {
    messageHistories[sessionId] = new InMemoryChatMessageHistory();
  }
  return messageHistories[sessionId];
}

async function runMemoryExample() {
  // 2. Define prompt with MessagesPlaceholder for history
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant."],
    new MessagesPlaceholder("chat_history"), // Past conversation gets inserted here
    ["human", "{input}"],                   // Current user input
  ]);

  // 3. Initialize model
  const model = await initChatModel("qwen3:1.7b", {
        modelProvider: "ollama",
        temperature: 0,
    });

  // 4. Build base LCEL chain
  const baseChain = prompt.pipe(model).pipe(new StringOutputParser());

  // 5. Wrap chain with RunnableWithMessageHistory
  const chainWithHistory = new RunnableWithMessageHistory({
    runnable: baseChain,
    getMessageHistory: (sessionId) => getMessageHistory(sessionId),
    inputMessagesKey: "input",
    historyMessagesKey: "chat_history",
  });

  // --- Session 1 Interaction ---
  const configUser1 = { configurable: { sessionId: "user-session-1" } };

  console.log("--- Turn 1 ---");
  const res1 = await chainWithHistory.invoke(
    { input: "Hi, my name is Santhosh and I live in Bengaluru." },
    configUser1
  );
  console.log("AI:", res1);

  console.log("\n--- Turn 2 (Asking for context from Turn 1) ---");
  const res2 = await chainWithHistory.invoke(
    { input: "What is my name and where do I live?" },
    configUser1
  );
  console.log("AI:", res2);

  // --- Session 2 Interaction (Isolated history) ---
  console.log("\n--- Separate Session Test ---");
  const configUser2 = { configurable: { sessionId: "user-session-2" } };

  const res3 = await chainWithHistory.invoke(
    { input: "What is my name?" },
    configUser2
  );
  console.log("AI (User 2):", res3);
}

runMemoryExample().catch(console.error);
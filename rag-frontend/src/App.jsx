import React, { useState, useRef, useEffect } from "react";
import "./App.css";

const API_BASE_URL = "http://localhost:3000/api";

export default function App() {
  // Ingestion State
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestStatus, setIngestStatus] = useState(null);

  // Chat State
  const [sessionId, setSessionId] = useState(() => `session-${Math.floor(1000 + Math.random() * 9000)}`);
  const [inputQuestion, setInputQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  // Auto-resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [inputQuestion]);

  // Direct Ingestion API Call
  const handleSyncDocs = async () => {
    setIsIngesting(true);
    setIngestStatus({ type: "info", text: "Syncing & vectorizing docs..." });

    try {
      const res = await fetch(`${API_BASE_URL}/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const data = await res.json();

      if (data.success) {
        setIngestStatus({
          type: "success",
          text: `Indexed ${data.chunksIndexed || 0} chunk(s) successfully`,
        });
      } else {
        setIngestStatus({ type: "error", text: data.message || "Ingestion failed" });
      }
    } catch (err) {
      setIngestStatus({ type: "error", text: "Connection error" });
    } finally {
      setIsIngesting(false);
      setTimeout(() => setIngestStatus(null), 4000);
    }
  };

  // Clear Chat Handler
  const handleClearChat = () => {
    if (isStreaming) return;
    setMessages([]);
    setSessionId(`session-${Math.floor(1000 + Math.random() * 9000)}`);
    setInputQuestion("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  // Handle SSE Streamed Chat Question
  const handleSendMessage = async (e) => {
    if (e) e.preventDefault();
    if (!inputQuestion.trim() || isStreaming) return;

    const userQuery = inputQuestion.trim();
    setInputQuestion("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    const userMsgId = Date.now();
    const assistantMsgId = userMsgId + 1;

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user", content: userQuery },
      { id: assistantMsgId, role: "assistant", content: "" },
    ]);

    setIsStreaming(true);

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userQuery, sessionId }),
      });

      if (!response.ok) throw new Error(`Server status ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let streamBuffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        streamBuffer += decoder.decode(value, { stream: true });
        const frames = streamBuffer.split("\n\n");
        streamBuffer = frames.pop() || "";

        for (const frame of frames) {
          if (frame.startsWith("data: ")) {
            const payload = frame.replace("data: ", "").trim();

            if (payload === "[DONE]") break;

            try {
              const parsed = JSON.parse(payload);
              if (parsed.chunk) {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMsgId
                      ? { ...msg, content: msg.content + parsed.chunk }
                      : msg
                  )
                );
              }
            } catch (err) {
              console.error("SSE Parsing error:", err);
            }
          }
        }
      }
    } catch (error) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMsgId
            ? { ...msg, content: `Error: ${error.message}` }
            : msg
        )
      );
    } finally {
      setIsStreaming(false);
    }
  };

  // Keyboard handler for Enter vs Shift+Enter
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="gemini-app">
      {/* Top Navbar */}
      <header className="navbar">
        <div className="nav-brand">
          <span className="gemini-logo">✦</span>
          <span className="brand-title">Gemini RAG Studio</span>
        </div>

        <div className="nav-actions">
          {ingestStatus && (
            <span className={`status-pill ${ingestStatus.type}`}>
              {ingestStatus.text}
            </span>
          )}

          <button
            className="secondary-btn"
            onClick={handleClearChat}
            disabled={isStreaming}
            title="Clear conversation and reset session"
          >
            Clear Chat
          </button>

          <button
            className="sync-btn"
            onClick={handleSyncDocs}
            disabled={isIngesting}
          >
            <span className={`sync-icon ${isIngesting ? "spinning" : ""}`}>↻</span>
            {isIngesting ? "Syncing..." : "Sync Knowledge Base"}
          </button>

          <span className="session-badge">{sessionId}</span>
        </div>
      </header>

      {/* Center Chat View */}
      <main className="chat-viewport">
        {messages.length === 0 ? (
          <div className="gemini-welcome">
            <h1 className="welcome-gradient">Hello, Developer</h1>
            <p className="welcome-sub">
              Your knowledge base is connected. Ask anything about your indexed documents.
            </p>
          </div>
        ) : (
          <div className="chat-stream">
            {messages.map((msg) => (
              <div key={msg.id} className={`message-row ${msg.role}`}>
                <div className="avatar">
                  {msg.role === "user" ? "U" : "✦"}
                </div>
                <div className="message-content">
                  {msg.content ? (
                    msg.content
                  ) : (
                    <span className="shimmer-text">Generating answer...</span>
                  )}
                </div>
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>
        )}

        {/* Floating Input Dock with Auto-resizing Textarea */}
        <div className="input-container">
          <div className="gemini-input-bar">
            <textarea
              ref={textareaRef}
              rows={1}
              placeholder="Ask Gemini about your knowledge base... (Press Enter to send, Shift+Enter for new line)"
              value={inputQuestion}
              onChange={(e) => setInputQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isStreaming}
            />
            <button
              type="button"
              onClick={handleSendMessage}
              className="send-btn"
              disabled={isStreaming || !inputQuestion.trim()}
            >
              ➔
            </button>
          </div>
          <div className="footer-caption">
            Gemini RAG can make mistakes. Verify important info.
          </div>
        </div>
      </main>
    </div>
  );
}
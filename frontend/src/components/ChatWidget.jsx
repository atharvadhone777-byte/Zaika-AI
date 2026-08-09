import { useState } from "react";
import { Send, Sparkles } from "lucide-react";
import { api } from "../api/client";

const SUGGESTED_QUESTIONS = [
  "How can I make this vegan?",
  "Can I cook this in an air fryer?",
  "How can I reduce calories?",
];

export default function ChatWidget({ recipeId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const ask = async (question) => {
    if (!question.trim() || loading) return;
    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setInput("");
    setLoading(true);
    try {
      const res = await api.chat(question, recipeId);
      setMessages((prev) => [...prev, { role: "assistant", text: res.answer, confidence: res.confidence }]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "assistant", text: `Something went wrong: ${err.message}`, confidence: "low" }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-surface-sunken rounded-card p-4 mt-6">
      <h4 className="flex items-center gap-2 font-display text-base mb-3">
        <Sparkles size={16} className="text-mustard" /> Ask about this recipe
      </h4>

      {messages.length === 0 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {SUGGESTED_QUESTIONS.map((q) => (
            <button
              key={q}
              onClick={() => ask(q)}
              className="text-xs bg-surface-raised text-ink-muted px-3 py-1.5 rounded-full
                         hover:text-mustard hover:bg-mustard-soft transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      <div className="space-y-2 mb-3 max-h-64 overflow-y-auto">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`text-sm rounded-card px-3 py-2 max-w-[90%] ${
              m.role === "user"
                ? "bg-mustard-soft text-mustard ml-auto"
                : "bg-surface-raised text-ink"
            }`}
          >
            {m.text}
          </div>
        ))}
        {loading && <div className="text-sm text-ink-muted px-3">Thinking...</div>}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
        className="flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g. Can I replace the butter?"
          className="flex-1 bg-surface-raised border border-white/10 rounded-card px-3 py-2 text-sm
                     text-ink placeholder:text-ink-muted focus:outline-none focus:ring-2 focus:ring-mustard/60"
        />
        <button
          type="submit"
          aria-label="Send"
          className="bg-mustard text-surface rounded-card px-3 py-2 hover:opacity-90 transition-opacity"
        >
          <Send size={16} />
        </button>
      </form>
    </div>
  );
}

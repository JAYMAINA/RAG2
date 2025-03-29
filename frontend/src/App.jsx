import { useState } from "react";
import axios from "axios";

export default function App() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");

  const handleAsk = async () => {
    if (!question) return;

    try {
      const res = await axios.post("http://127.0.0.1:8000/ask", {
        question,
      });
      setResponse(res.data.response);
    } catch (error) {
      setResponse("Error fetching response.");
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-6">
      <h1 className="text-3xl font-bold text-blue-600 mb-4">Ask Poa! AI</h1>
      <input
        type="text"
        className="w-full max-w-md p-2 border rounded-md shadow-md"
        placeholder="Type your question..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      />
      <button
        className="mt-4 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-500"
        onClick={handleAsk}
      >
        Ask
      </button>
      <div className="mt-4 p-4 bg-white shadow-md rounded-md w-full max-w-md">
        <p className="text-gray-700">{response}</p>
      </div>
    </div>
  );
}
      
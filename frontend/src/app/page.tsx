"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  Bot,
  FileText,
  Headphones,
  Home,
  Image as ImageIcon,
  Loader2,
  MessageSquare,
  Mic,
  Moon,
  Paperclip,
  Share2,
  ShieldCheck,
  Square,
  Sun,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface SchemeMatch {
  scheme_name?: string;
}

interface SourceLink {
  title: string;
  url: string;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  matches?: SchemeMatch[];
  citations?: string[];
  confidence?: number;
  sources?: SourceLink[];
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: number;
}

const INITIAL_ASSISTANT_MESSAGE =
  "Namaskaram! I am Vozhi, India’s Intelligent Benefits Orchestrator. How can I help you discover and unlock government schemes today?";

export default function VozhiApp() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [session, setSession] = useState("");
  const [history, setHistory] = useState<ChatSession[]>([]);
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [activeTab, setActiveTab] = useState<"home" | "chat">("home");
  const [uploadedDoc, setUploadedDoc] = useState<File | null>(null);
  const [showWhatsApp, setShowWhatsApp] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const createNewSession = useCallback(async (currentHistory: ChatSession[]) => {
    const newSessionId = `web-${Math.random().toString(36).substring(7)}`;
    const initialMsg: Message = {
      id: Date.now().toString(),
      role: "assistant",
      content: INITIAL_ASSISTANT_MESSAGE,
    };

    const newSession: ChatSession = {
      id: newSessionId,
      title: "New Conversation",
      messages: [initialMsg],
      updatedAt: Date.now(),
    };

    setHistory([newSession, ...currentHistory]);
    setSession(newSessionId);
    setMessages([initialMsg]);
    setActiveTab("chat");

    await fetch("http://127.0.0.1:8000/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: newSessionId,
        title: "New Conversation",
        messages: [initialMsg],
      }),
    }).catch(console.error);
  }, []);

  useEffect(() => {
    async function loadData() {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/sessions");
        if (!res.ok) return;

        const loadedHistory: ChatSession[] = await res.json();
        setHistory(loadedHistory);
        await createNewSession(loadedHistory);
      } catch (e) {
        console.error("Failed to load history from DB", e);
        await createNewSession([]);
      }
    }

    void loadData();
  }, [createNewSession]);

  useEffect(() => {
    const storedTheme = window.localStorage.getItem("vozhi-theme");
    const preferredTheme =
      storedTheme === "light" || storedTheme === "dark"
        ? storedTheme
        : window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
    setTheme(preferredTheme);
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    window.localStorage.setItem("vozhi-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (session && messages.length > 0) {
      const firstUser = messages.find((message) => message.role === "user");
      const newTitle = firstUser
        ? firstUser.content.length > 25
          ? `${firstUser.content.substring(0, 25)}...`
          : firstUser.content
        : "New Conversation";

      setHistory((prev) =>
        prev.map((chat) =>
          chat.id === session ? { ...chat, updatedAt: Date.now(), title: newTitle } : chat,
        ),
      );

      fetch("http://127.0.0.1:8000/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: session, title: newTitle, messages }),
      }).catch(console.error);
    }
  }, [messages, session]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    return () => {
      mediaRecorderRef.current?.stop();
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const startNewChat = async () => {
    await createNewSession(history);
  };

  const toggleTheme = () => {
    setTheme((currentTheme) => (currentTheme === "dark" ? "light" : "dark"));
  };

  const loadChat = async (chatId: string) => {
    setSession(chatId);
    setActiveTab("chat");
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/sessions/${chatId}`);
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages);
      }
    } catch (e) {
      console.error("Failed to load chat", e);
    }
  };

  const deleteChat = async (chatId: string) => {
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/sessions/${chatId}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        throw new Error("Failed to delete chat");
      }

      const remainingChats = history.filter((chat) => chat.id !== chatId);
      setHistory(remainingChats);

      if (session === chatId) {
        if (remainingChats.length > 0) {
          const nextChat = [...remainingChats].sort((a, b) => b.updatedAt - a.updatedAt)[0];
          await loadChat(nextChat.id);
        } else {
          setSession("");
          setMessages([]);
          setActiveTab("home");
        }
      }
    } catch (e) {
      console.error("Failed to delete chat", e);
    }
  };

  const streamChatResponse = async (
    request: RequestInit,
    userMessageContent: string,
    options?: { replaceUserMessageWithTranscript?: boolean },
  ) => {
    if (!session) return;

    const userId = Date.now().toString();
    const assistantId = `${userId}-assistant`;
    setMessages((prev) => [
      ...prev,
      {
        id: userId,
        role: "user",
        content: userMessageContent,
      },
      {
        id: assistantId,
        role: "assistant",
        content: "",
      },
    ]);
    setLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/chat/stream", request);
      if (!res.ok) throw new Error("API Error");
      if (!res.body) throw new Error("Streaming body unavailable");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);

          if (event.type === "transcript" && options?.replaceUserMessageWithTranscript) {
            setMessages((prev) =>
              prev.map((message) =>
                message.id === userId
                  ? {
                      ...message,
                      content: event.content || "Voice note",
                    }
                  : message,
              ),
            );
          }

          if (event.type === "chunk") {
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantId
                  ? { ...message, content: `${message.content}${event.content}` }
                  : message,
              ),
            );
          }

          if (event.type === "final") {
            const data = event.data;
            setMessages((prev) =>
              prev.map((message) => {
                if (message.id === userId && data.transcribed_text && options?.replaceUserMessageWithTranscript) {
                  return { ...message, content: data.transcribed_text };
                }
                if (message.id === assistantId) {
                  return {
                    ...message,
                    content: data.answer,
                    matches: data.matches,
                    confidence: data.confidence,
                    citations: data.citations || [],
                    sources: data.sources || [],
                  };
                }
                return message;
              }),
            );
          }

          if (event.type === "error") {
            throw new Error(event.detail || "Streaming error");
          }
        }
      }
    } catch {
      setMessages((prev) =>
        prev.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: "Sorry, I am having trouble connecting to the Vozhi Orchestrator right now.",
              }
            : message,
        ),
      );
    } finally {
      setLoading(false);
      setUploadedDoc(null);
    }
  };

  const handleSend = async () => {
    if (!input.trim() && !uploadedDoc) return;
    if (!session) return;

    const trimmedInput = input.trim();
    setInput("");

    if (uploadedDoc) {
      const formData = new FormData();
      formData.append("session_id", session);
      formData.append("query", trimmedInput);
      formData.append("file", uploadedDoc);
      await streamChatResponse(
        {
          method: "POST",
          body: formData,
        },
        trimmedInput || `(Uploaded Document: ${uploadedDoc.name})`,
      );
      return;
    }

    await streamChatResponse(
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: trimmedInput, session_id: session }),
      },
      trimmedInput,
    );
  };

  const handleToggleRecording = async () => {
    if (loading || !session) return;

    if (isRecording) {
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      audioChunksRef.current = [];

      const mimeType = MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "audio/mp4";
      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: recorder.mimeType || "audio/webm" });
        const extension = recorder.mimeType.includes("mp4") ? "m4a" : "webm";
        const audioFile = new File([audioBlob], `voice-note.${extension}`, {
          type: recorder.mimeType || "audio/webm",
        });
        mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
        audioChunksRef.current = [];

        const formData = new FormData();
        formData.append("session_id", session);
        formData.append("file", audioFile);
        void streamChatResponse(
          {
            method: "POST",
            body: formData,
          },
          "Transcribing your voice note...",
          { replaceUserMessageWithTranscript: true },
        );
      };

      recorder.start();
      setIsRecording(true);
    } catch {
      setIsRecording(false);
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-assistant-error`,
          role: "assistant",
          content: "Microphone access failed. Please allow mic access and try again, or send the query as text.",
        },
      ]);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  };

  const sortedHistory = [...history].sort((a, b) => b.updatedAt - a.updatedAt);

  return (
    <div className="flex h-screen w-full overflow-hidden bg-white font-sans text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      <aside className="hidden w-64 shrink-0 flex-col border-r border-zinc-200 bg-[#f9f9f9] dark:border-zinc-800 dark:bg-zinc-900 md:flex">
        <div className="flex items-center justify-between border-b border-zinc-200 px-6 py-5 dark:border-zinc-800">
          <h1 className="flex items-center gap-2 font-sans text-xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
            <ShieldCheck className="h-5 w-5 text-zinc-800 dark:text-zinc-100" />
            vozhi
          </h1>
        </div>

        <div className="flex-1 space-y-8 overflow-y-auto px-3 py-4">
          <div className="space-y-1">
            <button
              onClick={() => setActiveTab("home")}
              className={`flex w-full items-center gap-3 rounded-md px-3 py-2 text-left text-sm font-medium transition-colors ${
                activeTab === "home"
                  ? "bg-zinc-200/60 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100"
                  : "text-zinc-600 hover:bg-zinc-200/50 dark:text-zinc-400 dark:hover:bg-zinc-800/70"
              }`}
            >
              <Home className="h-4 w-4" /> Home
            </button>
          </div>

          <div className="flex min-h-0 flex-1 flex-col">
            <h3 className="px-3 pb-2 text-[11px] font-medium uppercase tracking-widest text-zinc-500 dark:text-zinc-400">
              Recent Chats
            </h3>
            <div className="custom-scrollbar mt-1 flex-1 space-y-1 overflow-y-auto pb-4">
              {sortedHistory.map((chat) => (
                <div
                  key={chat.id}
                  className={`group flex items-center gap-1 rounded-md transition-colors ${
                    session === chat.id && activeTab === "chat"
                      ? "bg-zinc-200/60 dark:bg-zinc-800"
                      : "hover:bg-zinc-200/50 dark:hover:bg-zinc-800/70"
                  }`}
                >
                  <button
                    onClick={() => void loadChat(chat.id)}
                    className={`flex min-w-0 flex-1 items-center gap-3 px-3 py-2 text-left text-sm font-medium ${
                      session === chat.id && activeTab === "chat"
                        ? "text-zinc-900 dark:text-zinc-100"
                        : "text-zinc-600 dark:text-zinc-400"
                    }`}
                  >
                    <MessageSquare className="h-4 w-4 shrink-0" />
                    <span className="truncate">{chat.title}</span>
                  </button>
                  <button
                    type="button"
                    aria-label={`Delete ${chat.title}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      void deleteChat(chat.id);
                    }}
                    className="mr-2 flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-zinc-400 opacity-0 transition hover:bg-white hover:text-red-600 group-hover:opacity-100 dark:hover:bg-zinc-700"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              ))}
              {history.length === 0 && (
                <div className="px-3 py-2 text-xs leading-relaxed text-zinc-400 dark:text-zinc-500">
                  No chats available. Click create new chat to start exploring about the
                  schemes.
                </div>
              )}
            </div>
          </div>
        </div>
      </aside>

      <main className="relative grid h-screen min-w-0 flex-1 grid-rows-[auto_1fr] bg-white dark:bg-zinc-950">
        <header className="z-10 flex items-center justify-between border-b border-zinc-100 bg-white px-8 py-5 dark:border-zinc-800 dark:bg-zinc-950">
          <div>
            <h2 className="text-xl font-semibold tracking-tight text-zinc-800 dark:text-zinc-100">
              {activeTab === "home" ? "Welcome Home" : "Vozhi Assistant"}
            </h2>
            <p className="mt-0.5 text-sm text-zinc-500 dark:text-zinc-400">
              {activeTab === "home"
                ? "Overview of indexed schemes and assistant capabilities."
                : "Discover eligible schemes, upload documents securely, and chat in any language."}
            </p>
          </div>
          <div className="flex gap-3">
            <Button
              variant="outline"
              onClick={() => setShowWhatsApp(true)}
              className="flex h-9 items-center gap-2 rounded-lg text-sm font-medium shadow-sm dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 dark:hover:bg-zinc-800"
            >
              <Share2 className="h-4 w-4" /> Try WhatsApp
            </Button>
            <Button
              variant="outline"
              onClick={() => void startNewChat()}
              className="flex h-9 items-center gap-2 rounded-lg text-sm font-medium shadow-sm dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 dark:hover:bg-zinc-800"
            >
              + New Chat
            </Button>
            <Button
              variant="ghost"
              onClick={toggleTheme}
              size="icon"
              className="h-9 w-9 rounded-lg text-zinc-500 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800"
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
          </div>
        </header>

        {activeTab === "home" ? (
          <div className="h-full w-full overflow-y-auto p-8 sm:p-12">
            <div className="mx-auto max-w-4xl space-y-8">
              <div className="rounded-3xl bg-gradient-to-br from-zinc-900 to-zinc-800 p-10 text-white shadow-xl">
                <h1 className="mb-3 text-3xl font-bold tracking-tight">Welcome to Vozhi.</h1>
                <p className="mb-8 max-w-2xl text-lg leading-relaxed text-zinc-300">
                  India&apos;s first Intelligent Government Benefits Orchestrator. Vozhi turns
                  natural voice and text queries into verified, bundled actionable benefits.
                </p>
                <div className="flex gap-4">
                  <Button
                    onClick={() => void startNewChat()}
                    className="h-11 rounded-xl border-0 bg-white px-6 font-medium text-zinc-900 shadow-sm hover:bg-zinc-100"
                  >
                    <MessageSquare className="mr-2 h-4 w-4" /> Create New Chat
                  </Button>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
                <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
                  <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-blue-50">
                    <FileText className="h-5 w-5 text-blue-600" />
                  </div>
                  <h3 className="mb-2 font-bold tracking-tight text-zinc-800 dark:text-zinc-100">
                    Graph RAG Engine
                  </h3>
                  <p className="text-sm leading-relaxed text-zinc-500 dark:text-zinc-400">
                    LlamaIndex PropertyGraph analyzes complex interdependencies between state
                    and central schemes, creating perfectly tailored eligibility bundles.
                  </p>
                </div>

                <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
                  <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-purple-50">
                    <Headphones className="h-5 w-5 text-purple-600" />
                  </div>
                  <h3 className="mb-2 font-bold tracking-tight text-zinc-800 dark:text-zinc-100">
                    Omnichannel Voice
                  </h3>
                  <p className="text-sm leading-relaxed text-zinc-500 dark:text-zinc-400">
                    Integrated with Sarvam AI and Twilio WhatsApp to ensure rural access.
                    Speak in any Indian language and get native audio responses.
                  </p>
                </div>

                <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
                  <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-green-50">
                    <ImageIcon className="h-5 w-5 text-green-600" />
                  </div>
                  <h3 className="mb-2 font-bold tracking-tight text-zinc-800 dark:text-zinc-100">
                    Document Intelligence
                  </h3>
                  <p className="text-sm leading-relaxed text-zinc-500 dark:text-zinc-400">
                    Upload Aadhaar or Income Certificates. EasyOCR and Llama 3.2 Vision
                    instantly extract exact demographic entities to prove eligibility.
                  </p>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <>
            <div className="custom-scrollbar h-full w-full overflow-y-auto px-4 pb-40 pt-8 sm:px-8">
              <div className="mx-auto max-w-[760px] space-y-8 pb-10">
                {messages.map((message) => (
                  <div key={message.id} className="flex w-full">
                    {message.role === "assistant" ? (
                      <div className="flex w-full flex-col">
                        <div className="mb-2 flex items-center gap-2 text-[13px] font-medium text-zinc-500 dark:text-zinc-400">
                          <Bot className="h-4 w-4 text-zinc-600 dark:text-zinc-300" />
                          <span>Vozhi Assistant</span>
                          {message.confidence && (
                            <>
                              <span className="mx-1 text-zinc-300 dark:text-zinc-700">|</span>
                              <span className="flex items-center gap-1.5 font-mono text-[11px] tracking-tight text-green-600">
                                <ShieldCheck className="h-3.5 w-3.5" /> Faithfulness:{" "}
                                {(message.confidence * 100).toFixed(1)}%
                              </span>
                            </>
                          )}
                        </div>
                        <div className="prose prose-sm ml-6 max-w-none whitespace-pre-wrap font-sans leading-relaxed text-zinc-800 dark:text-zinc-100">
                          {message.content}
                        </div>
                        {message.sources && message.sources.length > 0 && (
                          <div className="ml-6 mt-4 space-y-2">
                            <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                              Official Links
                            </p>
                            {message.sources.slice(0, 3).map((source, index) => (
                              <a
                                key={`${source.url}-${index}`}
                                href={source.url}
                                target="_blank"
                                rel="noreferrer"
                                className="block max-w-fit rounded-md border border-blue-200 bg-blue-50 px-3 py-2 text-sm text-blue-700 transition hover:bg-blue-100 dark:border-blue-900 dark:bg-blue-950/40 dark:text-blue-300 dark:hover:bg-blue-950/70"
                              >
                                {index + 1}. {source.title}
                              </a>
                            ))}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="flex w-full justify-end">
                        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-[#f0f0f0] px-4 py-2.5 text-[14.5px] leading-relaxed text-zinc-800 shadow-sm dark:bg-zinc-800 dark:text-zinc-100">
                          <p className="whitespace-pre-wrap">{message.content}</p>
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                {loading && (
                  <div className="flex justify-start gap-4">
                    <div className="flex w-full flex-col">
                      <div className="mb-2 flex items-center gap-2 text-[13px] font-medium text-zinc-500 dark:text-zinc-400">
                        <Bot className="h-4 w-4 animate-pulse text-zinc-600 dark:text-zinc-300" />
                        <span>Vozhi is thinking...</span>
                      </div>
                      <div className="ml-6 py-2">
                        <Loader2 className="h-5 w-5 animate-spin text-zinc-300 dark:text-zinc-600" />
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>

            <div className="pointer-events-none absolute bottom-6 left-0 z-20 flex w-full justify-center px-4">
              <div className="pointer-events-auto flex min-h-[56px] w-full max-w-[760px] flex-col items-center rounded-2xl border border-zinc-200 bg-white shadow-lg ring-1 ring-zinc-100 dark:border-zinc-800 dark:bg-zinc-900 dark:ring-zinc-800">
                {uploadedDoc && (
                  <div className="flex w-full items-center justify-between border-b border-zinc-100 px-4 pb-1 pt-3 dark:border-zinc-800">
                    <span className="flex items-center gap-1.5 rounded-md bg-zinc-100 px-2 py-1 text-xs font-semibold text-zinc-600 dark:bg-zinc-800 dark:text-zinc-200">
                      <FileText className="h-3.5 w-3.5 text-blue-600" /> {uploadedDoc.name}
                    </span>
                    <button
                      onClick={() => setUploadedDoc(null)}
                      className="text-[11px] font-semibold uppercase tracking-widest text-red-500 hover:text-red-700"
                    >
                      Remove
                    </button>
                  </div>
                )}

                <div className="relative flex w-full items-end gap-2 p-2">
                  <div className="flex shrink-0 pb-1 pl-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="relative h-8 w-8 rounded-full text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
                      disabled={loading || isRecording}
                    >
                      <input
                        type="file"
                        className="absolute inset-0 cursor-pointer opacity-0"
                        onChange={(e) => e.target.files && setUploadedDoc(e.target.files[0])}
                      />
                      <Paperclip className="h-4 w-4" />
                    </Button>
                  </div>

                  <Textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="What's on your mind? Type a query or upload an Aadhaar card..."
                    className="min-h-[40px] max-h-[160px] w-full flex-1 resize-none border-0 bg-transparent px-0 py-2.5 text-[14px] text-zinc-800 shadow-none placeholder:text-zinc-400 focus:outline-none focus:ring-0 focus-visible:ring-0 dark:text-zinc-100 dark:placeholder:text-zinc-500"
                    rows={1}
                  />

                  <div className="flex shrink-0 items-center gap-2 pb-1 pr-1">
                    <Button
                      onClick={() => void handleToggleRecording()}
                      disabled={loading || !session}
                      size="icon"
                      variant="ghost"
                      className={`h-8 w-8 rounded-full transition-all ${
                        isRecording
                          ? "bg-red-100 text-red-600 hover:bg-red-200 dark:bg-red-950 dark:text-red-300 dark:hover:bg-red-900"
                          : "text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
                      }`}
                    >
                      {isRecording ? <Square className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                    </Button>
                    <Button
                      onClick={() => void handleSend()}
                      disabled={loading || isRecording || !session || (!input.trim() && !uploadedDoc)}
                      size="icon"
                      className="h-8 w-8 rounded-full bg-zinc-600 text-white transition-all hover:bg-zinc-800 disabled:opacity-30 disabled:hover:bg-zinc-600 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300"
                    >
                      <ArrowUp className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </main>

      <Dialog open={showWhatsApp} onOpenChange={setShowWhatsApp}>
        <DialogContent className="max-w-md dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-100">
          <DialogHeader>
            <DialogTitle>Try Vozhi on WhatsApp</DialogTitle>
            <DialogDescription>
              Experience the power of Vozhi through Twilio WhatsApp. You can send voice
              notes or text anywhere in India.
            </DialogDescription>
          </DialogHeader>
          <div className="mt-4 space-y-3 rounded-xl bg-zinc-100 p-4 dark:bg-zinc-800">
            <p className="text-sm text-zinc-700 dark:text-zinc-200">
              1. Open your WhatsApp application.
            </p>
            <p className="text-sm text-zinc-700 dark:text-zinc-200">
              2. Send{" "}
              <span className="select-all rounded bg-zinc-200 px-1 py-0.5 font-mono text-xs dark:bg-zinc-700">
                join familiar-metal
              </span>{" "}
              to{" "}
              <span className="rounded bg-zinc-200 px-1 py-0.5 font-mono text-xs dark:bg-zinc-700">
                +1 415 523 8886
              </span>
            </p>
            <p className="text-sm text-zinc-700 dark:text-zinc-200">
              3. Ask a question like{" "}
              <em>&quot;I am a farmer from Bengal, what schemes am I eligible for?&quot;</em>
            </p>
          </div>
          <div className="mt-4 flex justify-end">
            <Button onClick={() => setShowWhatsApp(false)}>Done</Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

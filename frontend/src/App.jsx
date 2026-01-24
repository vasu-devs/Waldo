import React, { useState, useEffect, useRef } from 'react'
import { Send, ChevronRight, User, Upload, FileText, Loader2, Image as ImageIcon, Plus, CheckCircle2, XCircle, Trash2 } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import logo from './assets/logo.png'
import avatar from './assets/avatar.png'

const API_URL = "http://localhost:8000";

const Header = () => (
    <header className="glass-header">
        <div className="flex items-center gap-3">
            <div className="brand-accent">
                <img src={logo} alt="SOS 42" className="h-6 w-auto" />
            </div>
            <span className="font-bold text-lg tracking-tight">SOS 42</span>
        </div>

        <div className="hidden md:flex items-center gap-8">
            <a href="#" className="text-sm font-medium text-gray-500 hover:text-gray-900 transition-colors">Academy</a>
            <a href="#" className="text-sm font-medium text-gray-500 hover:text-gray-900 transition-colors">Features</a>
            <a href="#" className="text-sm font-medium text-gray-500 hover:text-gray-900 transition-colors">Pricing</a>
        </div>

        <button className="bg-[#6750A4] text-white px-5 py-2.5 rounded-xl font-semibold text-sm hover:bg-[#5A4590] transition-all flex items-center gap-2">
            Get on WhatsApp <ChevronRight size={16} />
        </button>
    </header>
)

const IngestionProgress = ({ filename }) => {
    const [status, setStatus] = useState("queued");
    const [message, setMessage] = useState("Starting upload...");
    const [progress, setProgress] = useState(0);
    const [stats, setStats] = useState({ current: 0, total: 0 });

    useEffect(() => {
        let pollInterval;

        const checkStatus = async () => {
            try {
                const res = await fetch(`${API_URL}/ingestion-status/${filename}`);
                const data = await res.json();

                setStatus(data.status);
                setMessage(data.message);

                if (data.current && data.total) {
                    setStats({ current: data.current, total: data.total });
                    if (data.progress) setProgress(data.progress);
                }

                if (data.status === "processing" && !data.progress) {
                    // Fallback fake progress if backend not reporting yet
                    setProgress(prev => Math.min(prev + 5, 90));
                } else if (data.status === "completed") {
                    setProgress(100);
                    clearInterval(pollInterval);
                } else if (data.status === "error") {
                    clearInterval(pollInterval);
                }
            } catch (e) {
                console.error("Polling error", e);
            }
        };

        pollInterval = setInterval(checkStatus, 1000);
        return () => clearInterval(pollInterval);
    }, [filename]);

    // Calculate Estimated Time (5s per item heuristic)
    const itemsLeft = stats.total - stats.current;
    const estTime = itemsLeft > 0 ? `${itemsLeft * 5}s` : "Calculating...";
    const isComplete = status === "completed";
    const isError = status === "error";

    // SVG Constants
    const radius = 18;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (progress / 100) * circumference;

    return (
        <div className="w-full bg-white border border-gray-200 rounded-xl p-4 mb-4 shadow-sm flex items-center gap-4 relative overflow-hidden">
            {/* Custom SVG Progress */}
            <div className="relative w-12 h-12 flex-shrink-0 flex items-center justify-center">
                <svg className="transform -rotate-90 w-12 h-12">
                    <circle
                        cx="24" cy="24" r={radius}
                        stroke="#F3F0FF" strokeWidth="4" fill="transparent"
                    />
                    {!isError && (
                        <circle
                            cx="24" cy="24" r={radius}
                            stroke="#6750A4" strokeWidth="4" fill="transparent"
                            strokeDasharray={circumference}
                            strokeDashoffset={offset}
                            strokeLinecap="round"
                            className="transition-all duration-500 ease-out"
                        />
                    )}
                </svg>
                <div className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-[#6750A4]">
                    {isError ? <XCircle size={16} className="text-red-500" /> : isComplete ? <CheckCircle2 size={16} className="text-green-500" /> : `${Math.round(progress)}%`}
                </div>
            </div>

            <div className="flex-1 min-w-0 z-10">
                <div className="flex justify-between items-start">
                    <h4 className="font-semibold text-sm text-gray-900 truncate">{filename}</h4>
                    {status === "processing" && stats.total > 0 && (
                        <span className="text-[10px] font-medium bg-purple-50 text-purple-700 px-2 py-0.5 rounded-full">
                            Est: ~{estTime}
                        </span>
                    )}
                </div>
                <p className="text-xs text-gray-500 mt-0.5 truncate">{message}</p>
                {status === "processing" && (
                    <div className="mt-1 flex items-center gap-2">
                        <div className="flex space-x-1">
                            {[...Array(3)].map((_, i) => (
                                <motion.div
                                    key={i}
                                    className="w-1 h-1 bg-purple-400 rounded-full"
                                    animate={{ opacity: [0.3, 1, 0.3] }}
                                    transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                                />
                            ))}
                        </div>
                        <span className="text-[10px] text-gray-400">Processing...</span>
                    </div>
                )}
            </div>

            {/* Background Decoration */}
            {status === "processing" && (
                <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-transparent via-purple-50/20 to-transparent"
                    animate={{ x: ['-100%', '200%'] }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                />
            )}
        </div>
    );
};

const Message = ({ isBot, text, documents, ingestionFile }) => (
    <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ type: "spring", damping: 20, stiffness: 100 }}
        className={`flex w-full ${isBot ? 'justify-start' : 'justify-end'} mb-6 items-start gap-4`}
    >
        {isBot && (
            <div className="w-10 h-10 rounded-full bg-[#F3F0FF] p-1 flex-shrink-0 border border-[#E9E4FF]">
                <img src={avatar} alt="AI" className="w-full h-full object-contain rounded-full" />
            </div>
        )}

        <div className={`max-w-[80%] sm:max-w-[70%] ${isBot ? 'bot-bubble' : 'user-bubble'}`}>
            {ingestionFile ? (
                <IngestionProgress filename={ingestionFile} />
            ) : (
                <div className="prose prose-sm max-w-none">
                    <ReactMarkdown>{text}</ReactMarkdown>
                </div>
            )}

            {/* Render only images inline - no citations */}
            {documents && documents.filter(doc => doc.original_image_path && doc.element_type === 'figure').length > 0 && (
                <div className="mt-4">
                    {documents.filter(doc => doc.original_image_path && doc.element_type === 'figure').slice(0, 1).map((doc, idx) => (
                        <div key={idx} className="rounded-lg overflow-hidden border border-gray-200 bg-white shadow-sm">
                            <img
                                src={`http://localhost:8000/images/${doc.original_image_path.split(/[/\\]/).pop()}`}
                                alt={`Figure from Page ${doc.page_number}`}
                                className="w-full h-auto object-contain"
                                onError={(e) => {
                                    e.target.parentElement.style.display = 'none';
                                }}
                            />
                        </div>
                    ))}
                </div>
            )}

        </div>

        {!isBot && (
            <div className="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center flex-shrink-0 text-gray-500">
                <User size={20} />
            </div>
        )}
    </motion.div>
)

const LoadingState = () => (
    <div className="flex justify-start mb-6 gap-4">
        <div className="w-10 h-10 rounded-full bg-[#F3F0FF] p-1 flex-shrink-0 animate-pulse" />
        <div className="bg-white border border-gray-100 rounded-2xl rounded-tl-none p-4 flex gap-1.5 items-center shadow-sm">
            <div className="w-1.5 h-1.5 bg-[#6750A4] rounded-full animate-bounce [animation-delay:-0.3s]" />
            <div className="w-1.5 h-1.5 bg-[#6750A4] rounded-full animate-bounce [animation-delay:-0.15s]" />
            <div className="w-1.5 h-1.5 bg-[#6750A4] rounded-full animate-bounce" />
        </div>
    </div>
)

export default function App() {
    const [messages, setMessages] = useState([
        { text: "Hello! I'm your **Multimodal RAG Assistant**. Upload a PDF to get started, then ask me anything about it!", isBot: true }
    ])
    const [input, setInput] = useState("")
    const [isThinking, setIsThinking] = useState(false)
    const [isUploading, setIsUploading] = useState(false)
    const [isResetting, setIsResetting] = useState(false)
    const scrollRef = useRef(null)
    const fileInputRef = useRef(null)

    useEffect(() => {
        scrollRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages, isThinking])

    const handleSend = async () => {
        if (!input.trim()) return

        const userMsg = { text: input, isBot: false }
        setMessages(prev => [...prev, userMsg])
        setInput("")
        setIsThinking(true)

        try {
            const response = await fetch(`${API_URL}/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: userMsg.text })
            });

            if (!response.ok) throw new Error("Failed to fetch response");

            const data = await response.json();

            setMessages(prev => [...prev, {
                text: data.response,
                documents: data.documents,
                isBot: true
            }]);
        } catch (error) {
            console.error("Chat error:", error);
            setMessages(prev => [...prev, {
                text: "**Error:** Could not connect to the AI. Is the backend running?",
                isBot: true
            }]);
        } finally {
            setIsThinking(false);
        }
    }

    const triggerUpload = () => {
        fileInputRef.current?.click();
    }

    const handleReset = async () => {
        if (!confirm("Are you sure? This will clear all uploaded documents and chat history.")) {
            return;
        }

        setIsResetting(true);

        try {
            const response = await fetch("http://localhost:8000/reset", {
                method: "DELETE",
            });

            if (!response.ok) {
                throw new Error(await response.text());
            }

            const data = await response.json();
            console.log("Reset response:", data);

            // Clear local chat state
            setMessages([{
                text: "🔄 **System Reset Complete!** Ready for a new document.",
                isBot: true
            }]);
            setInput("");

        } catch (error) {
            console.error("Reset error:", error);
            setMessages(prev => [...prev, {
                text: `❌ **Reset Error:** ${error.message}`,
                isBot: true
            }]);
        } finally {
            setIsResetting(false);
        }
    }

    const handleFileChange = async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setIsUploading(true);
        const formData = new FormData();
        formData.append("file", file);

        try {
            // Add a temporary "system" message for the upload progress
            setMessages(prev => [...prev, {
                isBot: true,
                ingestionFile: file.name
            }]);

            // Explicit backend URL, no Content-Type header (browser handles multipart boundary)
            const response = await fetch("http://localhost:8000/ingest", {
                method: "POST",
                body: formData
            });

            // Strict error check: throw if not 200 OK
            if (!response.ok) {
                throw new Error(await response.text());
            }

            // Parse and log server response for debugging
            const data = await response.json();
            console.log("Ingest server response:", data);

            // Success: IngestionProgress component will handle polling updates
            setMessages(prev => [...prev, {
                text: `✅ **Ingestion complete!** File "${file.name}" has been processed.`,
                isBot: true
            }]);

        } catch (error) {
            console.error("Upload error:", error);
            // Surface error in UI
            setMessages(prev => [...prev, {
                text: `❌ **Upload Error:** ${error.message}`,
                isBot: true
            }]);
        } finally {
            setIsUploading(false);
            if (fileInputRef.current) fileInputRef.current.value = "";
        }
    }

    return (
        <div className="min-h-screen">
            <Header />

            {/* Hidden File Input */}
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="application/pdf"
                className="hidden"
            />

            <main className="chat-container">
                <AnimatePresence>
                    {messages.map((m, i) => (
                        <Message key={i} {...m} />
                    ))}
                    {isThinking && <LoadingState />}
                </AnimatePresence>
                <div ref={scrollRef} />
            </main>

            <div className="input-wrapper">
                <div className="relative flex items-center gap-2">
                    <button
                        onClick={triggerUpload}
                        disabled={isUploading}
                        className="w-10 h-10 rounded-full flex items-center justify-center bg-gray-100 text-gray-500 hover:bg-[#F3F0FF] hover:text-[#6750A4] transition-colors flex-shrink-0"
                        title="Upload PDF"
                    >
                        {isUploading ? <Loader2 size={20} className="animate-spin" /> : <Plus size={24} />}
                    </button>

                    <button
                        onClick={handleReset}
                        disabled={isResetting}
                        className="w-10 h-10 rounded-full flex items-center justify-center bg-red-50 text-red-500 hover:bg-red-100 transition-colors flex-shrink-0"
                        title="Reset System"
                    >
                        {isResetting ? <Loader2 size={18} className="animate-spin" /> : <Trash2 size={18} />}
                    </button>

                    <div className="relative flex-1">
                        <input
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                            placeholder="Ask a question about your documents..."
                            className="input-field"
                            style={{ paddingLeft: '1rem' }}
                            disabled={isThinking}
                        />
                        <button onClick={handleSend} disabled={isThinking} className="btn-send absolute right-2 top-1/2 -translate-y-1/2">
                            <Send size={20} />
                        </button>
                    </div>
                </div>
                <p className="text-center mt-4 text-[11px] text-gray-400 font-medium">
                    Powered by Multimodal RAG • SOS 42 AI Engine
                </p>
            </div>
        </div>
    )
}

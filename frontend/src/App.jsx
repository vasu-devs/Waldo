import React, { useState, useEffect, useRef } from 'react'
import { Send, Zap, ChevronRight, User, Bot, Sparkles, Image as ImageIcon } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import logo from './assets/logo.png'
import avatar from './assets/avatar.png'

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

const Message = ({ isBot, text, image, imageAlt }) => (
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
            <div className="prose prose-sm max-w-none">
                <ReactMarkdown>{text}</ReactMarkdown>
            </div>

            {image && (
                <div className="multimodal-container bg-gray-50 p-2 mt-4">
                    <img src={image} alt={imageAlt} className="w-full h-auto rounded-lg" />
                    {imageAlt && (
                        <div className="mt-2 px-1 flex items-center gap-2 text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
                            <Sparkles size={12} className="text-[#6750A4]" /> {imageAlt}
                        </div>
                    )}
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
        { text: "Hello! I'm your **Multimodal RAG Assistant**. I can help you analyze documents, visualize data, and answer complex queries with precision.\n\nHow can I support your research today?", isBot: true }
    ])
    const [input, setInput] = useState("")
    const [isThinking, setIsThinking] = useState(false)
    const scrollRef = useRef(null)

    useEffect(() => {
        scrollRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages, isThinking])

    const handleSend = () => {
        if (!input.trim()) return

        const userMsg = { text: input, isBot: false }
        setMessages(prev => [...prev, userMsg])
        setInput("")
        setIsThinking(true)

        // Mock API response
        setTimeout(() => {
            setMessages(prev => [...prev, {
                text: "I've analyzed the financial report as requested. Based on the quarterly projections, we're seeing strong growth in operational efficiency. \n\nHere is the **Resource Allocation Map** derived from the source documents:",
                image: "https://images.unsplash.com/photo-1551288049-bbbda536ad89?auto=format&fit=crop&q=80&w=1000",
                imageAlt: "Quarterly Growth Vector Analysis",
                isBot: true
            }])
            setIsThinking(false)
        }, 1800)
    }

    return (
        <div className="min-h-screen">
            <Header />

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
                <div className="relative">
                    <input
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="Ask a question about your documents..."
                        className="input-field"
                    />
                    <button onClick={handleSend} className="btn-send">
                        <Send size={20} />
                    </button>
                </div>
                <p className="text-center mt-4 text-[11px] text-gray-400 font-medium">
                    Powered by Multimodal RAG • SOS 42 AI Engine
                </p>
            </div>
        </div>
    )
}

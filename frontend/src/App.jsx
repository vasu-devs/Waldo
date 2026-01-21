import React from 'react'
import { Search, Send, Zap, Clock, Cloud } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'

const Header = () => (
    <header className="glass-nav">
        <div className="flex items-center gap-2">
            <div className="w-10 h-10 bg-brand-navy rounded-lg flex items-center justify-center text-white font-black text-xl shadow-sos">
                42
            </div>
            <span className="font-extrabold text-xl tracking-tight">SOS 42</span>
        </div>
        <nav className="hidden md:flex items-center gap-8 font-semibold text-sm uppercase tracking-wider">
            <a href="#" className="hover:text-brand-purple border-b-2 border-brand-yellow hover:border-brand-purple transition-all pb-1">Academy</a>
            <a href="#" className="hover:text-brand-purple border-b-2 border-brand-yellow hover:border-brand-purple transition-all pb-1">Features</a>
            <a href="#" className="hover:text-brand-purple border-b-2 border-brand-yellow hover:border-brand-purple transition-all pb-1">Pricing</a>
        </nav>
        <button className="btn-primary flex items-center gap-2 text-sm">
            Get Started <Zap size={16} fill="white" />
        </button>
    </header>
)

const Footer = () => (
    <footer className="py-12 bg-brand-black text-white px-6 mt-auto">
        <div className="max-w-4xl mx-auto flex flex-col md:flex-row justify-between items-center gap-8">
            <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-white rounded-md flex items-center justify-center text-brand-black font-black text-lg">
                    42
                </div>
                <span className="font-bold text-lg">SOS 42</span>
            </div>
            <div className="flex gap-4">
                <div className="bg-white/10 px-4 py-2 rounded-lg border border-white/20">App Store</div>
                <div className="bg-white/10 px-4 py-2 rounded-lg border border-white/20">Google Play</div>
            </div>
            <p className="text-gray-400 text-sm">© 2026 SOS 42. Adulting simplified.</p>
        </div>
    </footer>
)

const ImageCard = ({ src, alt }) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="sos-card overflow-hidden my-4 max-w-md"
    >
        <img src={src} alt={alt} className="w-full h-auto object-cover hover:scale-105 transition-transform duration-500 cursor-zoom-in" />
        {alt && <div className="p-3 bg-white border-t-2 border-brand-black text-xs font-medium italic">{alt}</div>}
    </motion.div>
)

const ChatMessage = ({ message, isBot }) => (
    <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ type: "spring", damping: 20, stiffness: 100 }}
        className={`flex w-full ${isBot ? 'justify-start' : 'justify-end'} mb-4`}
    >
        <div className={`max-w-[85%] ${isBot ? 'bg-white rounded-2xl rounded-tl-none border-2 border-brand-black' : 'bg-brand-purple text-white rounded-2xl rounded-tr-none shadow-sos'} p-4`}>
            {isBot ? (
                <div className="prose prose-sm font-medium">
                    <ReactMarkdown>{message.text}</ReactMarkdown>
                    {message.image && <ImageCard src={message.image} alt={message.imageAlt} />}
                </div>
            ) : (
                <p className="font-semibold">{message.text}</p>
            )}
        </div>
    </motion.div>
)

const ThinkingAnimation = () => (
    <div className="flex gap-2 p-4 bg-white border-2 border-brand-black rounded-2xl rounded-tl-none w-fit mb-4">
        <motion.div
            animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
            transition={{ repeat: Infinity, duration: 1 }}
            className="w-2 h-4 bg-brand-purple rounded-full"
        />
        <motion.div
            animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
            transition={{ repeat: Infinity, duration: 1, delay: 0.2 }}
            className="w-2 h-4 bg-brand-pink rounded-full"
        />
        <motion.div
            animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
            transition={{ repeat: Infinity, duration: 1, delay: 0.4 }}
            className="w-2 h-4 bg-brand-yellow rounded-full"
        />
    </div>
)

export default function App() {
    const [messages, setMessages] = React.useState([
        { text: "Hey! I'm your RAG assistant. How can I help you adult today?", isBot: true }
    ])
    const [input, setInput] = React.useState("")
    const [isThinking, setIsThinking] = React.useState(false)
    const chatEndRef = React.useRef(null)

    const scrollToBottom = () => chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
    React.useEffect(scrollToBottom, [messages, isThinking])

    const handleSend = () => {
        if (!input.trim()) return
        const userMsg = { text: input, isBot: false }
        setMessages(prev => [...prev, userMsg])
        setInput("")
        setIsThinking(true)

        // Mock response after delay
        setTimeout(() => {
            const botMsg = {
                text: "Based on your documents, here's a visualization of your progress. Adulting is a marathon, not a sprint! \n\n### Key Metrics:\n- Budget adherence: **85%**\n- Chores completed: **12/15**",
                image: "https://via.placeholder.com/600x400/6750A4/FFFFFF?text=Multimodal+Chart+Example",
                imageAlt: "Adulting Progress Chart",
                isBot: true
            }
            setMessages(prev => [...prev, botMsg])
            setIsThinking(false)
        }, 2000)
    }

    return (
        <div className="flex flex-col min-h-screen">
            <Header />

            <main className="flex-grow pt-28 pb-32 px-4">
                <div className="max-w-3xl mx-auto">
                    <AnimatePresence>
                        {messages.map((m, i) => (
                            <ChatMessage key={i} message={m} isBot={m.isBot} />
                        ))}
                        {isThinking && <ThinkingAnimation />}
                    </AnimatePresence>
                    <div ref={chatEndRef} />
                </div>
            </main>

            <div className="fixed bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-brand-yellow via-brand-yellow to-transparent">
                <div className="max-w-3xl mx-auto flex gap-3 p-3 bg-white border-2 border-brand-black rounded-pill shadow-pill">
                    <input
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="Ask anything about your documents..."
                        className="flex-grow bg-transparent outline-none px-4 font-medium"
                    />
                    <button
                        onClick={handleSend}
                        className="bg-brand-purple text-white p-3 rounded-full hover:bg-brand-navy transition-colors"
                    >
                        <Send size={20} />
                    </button>
                </div>
            </div>

            <Footer />
        </div>
    )
}

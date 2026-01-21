import React from 'react'
import { Send, Zap, ChevronRight } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import logo from './assets/logo.png'
import avatar from './assets/avatar.png'
import doodles from './assets/doodles.png'

const Header = () => (
    <header className="glass-nav">
        <div className="flex items-center gap-3">
            <img src={logo} alt="SOS 42 Logo" className="h-10 w-auto" />
            <span className="font-extrabold text-2xl tracking-tighter hidden sm:block">SOS 42</span>
        </div>
        <nav className="hidden md:flex items-center gap-10 font-bold text-sm uppercase tracking-widest text-brand-black/70">
            <a href="#" className="hover:text-brand-purple transition-colors">Academy</a>
            <a href="#" className="hover:text-brand-purple transition-colors">Features</a>
            <a href="#" className="hover:text-brand-purple transition-colors">Pricing</a>
        </nav>
        <button className="btn-primary flex items-center gap-2 text-sm">
            Get on WhatsApp <ChevronRight size={18} strokeWidth={3} />
        </button>
    </header>
)

const Footer = () => (
    <footer className="py-16 bg-brand-black text-white px-6 mt-20 relative overflow-hidden">
        <div className="max-w-5xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-10 relative z-10">
            <div className="flex flex-col gap-4">
                <div className="flex items-center gap-3">
                    <img src={logo} alt="SOS 42" className="h-8 invert grayscale" />
                    <span className="font-black text-2xl tracking-tighter">SOS 42</span>
                </div>
                <p className="text-gray-400 max-w-xs font-medium">Adulting simplified for students. Get the aid you need, when you need it.</p>
            </div>
            <div className="flex flex-wrap gap-6">
                <div className="bg-white/5 hover:bg-white/10 transition-colors px-6 py-3 rounded-xl border border-white/10 flex items-center gap-3 cursor-pointer">
                    <div className="w-6 h-6 bg-white rounded-full" />
                    <div className="flex flex-col">
                        <span className="text-[10px] uppercase font-bold opacity-60">Download on</span>
                        <span className="font-bold">App Store</span>
                    </div>
                </div>
                <div className="bg-white/5 hover:bg-white/10 transition-colors px-6 py-3 rounded-xl border border-white/10 flex items-center gap-3 cursor-pointer">
                    <div className="w-6 h-6 bg-white rounded-full" />
                    <div className="flex flex-col">
                        <span className="text-[10px] uppercase font-bold opacity-60">Get it on</span>
                        <span className="font-bold">Google Play</span>
                    </div>
                </div>
            </div>
        </div>
        <div className="max-w-5xl mx-auto mt-16 pt-8 border-t border-white/10 flex flex-col sm:flex-row justify-between gap-4 text-xs font-bold text-gray-500 uppercase tracking-widest">
            <div className="flex gap-6">
                <a href="#" className="hover:text-white transition-colors">Privacy</a>
                <a href="#" className="hover:text-white transition-colors">Terms</a>
            </div>
            <p>© 2026 SOS 42. Made with ❤️ for Students.</p>
        </div>
    </footer>
)

const ImageCard = ({ src, alt }) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="sos-card overflow-hidden my-6 max-w-lg"
    >
        <div className="relative group">
            <img src={src} alt={alt} className="w-full h-auto object-cover group-hover:scale-105 transition-transform duration-700 cursor-zoom-in" />
            <div className="absolute inset-0 bg-brand-purple/10 opacity-0 group-hover:opacity-100 transition-opacity" />
        </div>
        {alt && <div className="p-4 bg-white border-t-2 border-brand-black text-sm font-bold uppercase tracking-wide">{alt}</div>}
    </motion.div>
)

const ChatMessage = ({ message, isBot }) => (
    <motion.div
        initial={{ opacity: 0, x: isBot ? -20 : 20, y: 20 }}
        animate={{ opacity: 1, x: 0, y: 0 }}
        transition={{ type: "spring", damping: 25, stiffness: 120 }}
        className={`flex w-full ${isBot ? 'justify-start' : 'justify-end'} mb-10 items-end gap-3`}
    >
        {isBot && (
            <div className="w-10 h-10 rounded-full border-2 border-brand-black bg-brand-yellow flex-shrink-0 overflow-hidden shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]">
                <img src={avatar} alt="Bot Avatar" className="w-full h-full object-cover" />
            </div>
        )}
        <div className={`max-w-[80%] ${isBot ? 'chat-bubble-bot' : 'chat-bubble-user'}`}>
            {isBot ? (
                <div className="prose prose-sm font-bold leading-relaxed text-brand-black">
                    <ReactMarkdown>{message.text}</ReactMarkdown>
                    {message.image && <ImageCard src={message.image} alt={message.imageAlt} />}
                </div>
            ) : (
                <p className="font-black text-lg">{message.text}</p>
            )}
        </div>
        {!isBot && (
            <div className="w-10 h-10 rounded-full border-2 border-brand-black bg-white flex-shrink-0 flex items-center justify-center font-black text-xs shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]">
                ME
            </div>
        )}
    </motion.div>
)

const ThinkingAnimation = () => (
    <div className="flex gap-2 p-5 bg-white border-2 border-brand-black rounded-2xl rounded-tl-none w-fit mb-10 shadow-[4px_4px_0px_0px_#FF4D6D]">
        {[0, 0.2, 0.4].map((delay) => (
            <motion.div
                key={delay}
                animate={{
                    y: [0, -8, 0],
                    backgroundColor: ["#6750A4", "#FF4D6D", "#FBBF24", "#6750A4"]
                }}
                transition={{ repeat: Infinity, duration: 1.2, delay }}
                className="w-3 h-3 border-2 border-brand-black rounded-full"
            />
        ))}
    </div>
)

export default function App() {
    const [messages, setMessages] = React.useState([
        { text: "Yo! I'm your SOS 42 AI agent. Drop your docs here and let's crush some tasks together! 🚀", isBot: true }
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

        setTimeout(() => {
            const botMsg = {
                text: "Checked your progress! You're absolutely killing it. Here's how your stats look relative to your goals. \n\n### 📈 Real-time Analytics:\n*   Productivity Score: **92%**\n*   Focus Sessions: **8/10**\n*   Deadline Risk: **Low**",
                image: "https://via.placeholder.com/800x500/6750A4/FFFFFF?text=SOS+42+Analytics+Report",
                imageAlt: "Student Dashboard Overview",
                isBot: true
            }
            setMessages(prev => [...prev, botMsg])
            setIsThinking(false)
        }, 2000)
    }

    return (
        <div className="flex flex-col min-h-screen selection:bg-brand-pink selection:text-white">
            {/* Decorative Doodles */}
            <img src={doodles} className="doodle-bg top-40 -left-20 w-64 rotate-12" alt="" />
            <img src={doodles} className="doodle-bg bottom-80 -right-20 w-80 -rotate-12" alt="" />

            <Header />

            <main className="flex-grow pt-40 pb-48 px-4 relative z-10">
                <div className="max-w-4xl mx-auto">
                    <AnimatePresence initial={false}>
                        {messages.map((m, i) => (
                            <ChatMessage key={i} message={m} isBot={m.isBot} />
                        ))}
                        {isThinking && <ThinkingAnimation />}
                    </AnimatePresence>
                    <div ref={chatEndRef} />
                </div>
            </main>

            <div className="fixed bottom-0 left-0 right-0 p-8 z-50 bg-gradient-to-t from-brand-yellow via-brand-yellow/80 to-transparent">
                <div className="input-container group focus-within:translate-y-[-4px] focus-within:shadow-[0px_12px_0px_0px_rgba(0,0,0,1)] transition-all">
                    <input
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="What's the plan for today? Ask SOS 42..."
                        className="flex-grow bg-transparent outline-none px-4 py-4 font-bold text-lg placeholder:text-brand-black/30"
                    />
                    <button
                        onClick={handleSend}
                        className="bg-brand-purple text-white p-4 rounded-full border-2 border-brand-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:scale-110 active:scale-95 transition-all ml-2"
                    >
                        <Send size={24} strokeWidth={3} />
                    </button>
                </div>
            </div>

            <Footer />
        </div>
    )
}

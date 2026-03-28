import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare, X, Send } from 'lucide-react';

const ChatbotWidget = ({ language }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  
  // Set default welcoming message whenever language changes
  useEffect(() => {
    let greeting = "Hello! I'm your LeafGuard AI assistant. Ask me anything about crop diseases, remedies, or agriculture in general.";
    if (language === 'hi') greeting = "नमस्ते! मैं आपका लीफगार्ड एआई सहायक हूं। मुझसे फसल रोगों, उपचार या कृषि के बारे में कुछ भी पूछें।";
    if (language === 'mr') greeting = "नमस्कार! मी तुमचा लीफगार्ड एआय सहाय्यक आहे. मला पीक रोग, उपाय किंवा शेतीबद्दल काहीही विचारा.";
    
    // Reset or update first message
    setMessages([{ id: 1, text: greeting, sender: 'bot' }]);
  }, [language]);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isOpen]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { id: Date.now(), text: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInput('');

    try {
      const response = await fetch(`/chat?lang=${language}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage.text })
      });
      const data = await response.json();
      
      setMessages(prev => [...prev, { id: Date.now()+1, text: data.reply, sender: 'bot' }]);
    } catch (error) {
      console.error(error);
      let errStr = "I'm sorry, I cannot connect to the server right now.";
      if (language === 'hi') errStr = "मुझे खेद है, मैं अभी सर्वर से कनेक्ट नहीं हो सकता।";
      if (language === 'mr') errStr = "क्षमस्व, मी आत्ता सर्व्हरशी कनेक्ट होऊ शकत नाही.";
      setMessages(prev => [...prev, { id: Date.now()+1, text: errStr, sender: 'bot' }]);
    }
  };

  let title = "LeafGuard Assistant";
  let placeholder = "Ask a question...";
  if (language === 'hi') {
    title = "लीफगार्ड सहायक";
    placeholder = "एक सवाल पूछें...";
  } else if (language === 'mr') {
    title = "लीफगार्ड सहाय्यक";
    placeholder = "एक प्रश्न विचारा...";
  }

  return (
    <div className="chatbot-widget">
      {isOpen ? (
        <div className="chat-window">
          <div className="chat-header">
            <span>{title}</span>
            <button onClick={() => setIsOpen(false)} style={{background:'transparent', border:'none', color:'white', cursor:'pointer'}}>
              <X size={20} />
            </button>
          </div>
          
          <div className="chat-messages">
            {messages.map((msg) => (
              <div key={msg.id} className={`message ${msg.sender}`}>
                {msg.text}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <form className="chat-input" onSubmit={handleSend}>
            <input 
              type="text" 
              placeholder={placeholder}
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button type="submit" className="send-btn" disabled={!input.trim()}>
              <Send size={18} />
            </button>
          </form>
        </div>
      ) : (
        <button className="chat-fab" onClick={() => setIsOpen(true)}>
          <MessageSquare size={24} />
        </button>
      )}
    </div>
  );
};

export default ChatbotWidget;

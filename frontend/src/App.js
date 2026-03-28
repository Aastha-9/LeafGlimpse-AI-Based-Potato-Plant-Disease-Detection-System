import React, { useState } from "react";
import "./App.css";

import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import DiseaseClassifier from "./components/DiseaseClassifier";
import BlogSection from "./components/BlogSection";
import ChatbotWidget from "./components/ChatbotWidget";

function App() {
  const [language, setLanguage] = useState("en");

  return (
    <div className="app-container">
      <Navbar language={language} setLanguage={setLanguage} />
      
      <main className="content">
        <Hero language={language} />
        <DiseaseClassifier language={language} />
        <BlogSection language={language} />
      </main>

      <ChatbotWidget language={language} />
    </div>
  );
}

export default App;

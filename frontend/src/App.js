import { useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [language, setLanguage] = useState("en");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const predict = async () => {
    if (!image) return alert("Please upload an image");

    setLoading(true);
    const formData = new FormData();
    formData.append("file", image);

    const response = await fetch(
      `http://127.0.0.1:8000/predict?lang=${language}`,
      {
        method: "POST",
        body: formData,
      }
    );

    const data = await response.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div className="container">
      <h1>🌿 Potato Leaf Disease Detection</h1>

      <select onChange={(e) => setLanguage(e.target.value)}>
        <option value="en">English</option>
        <option value="hi">Hindi</option>
        <option value="mr">Marathi</option>
      </select>

      <input type="file" onChange={handleImageChange} />

      {preview && <img src={preview} alt="preview" className="preview" />}

      <button onClick={predict}>
        {loading ? "Predicting..." : "Predict"}
      </button>

      {result && (
        <div className="result-card">
          <h2>{result.message}</h2>
          <p>Confidence: {result.confidence}%</p>

          <h3>Recommendations:</h3>
          <ul>
            {result.recommendations.map((rec, index) => (
              <li key={index}>{rec}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;

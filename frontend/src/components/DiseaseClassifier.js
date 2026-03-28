import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, Camera, Loader2, AlertTriangle, CheckCircle, X } from 'lucide-react';

const DiseaseClassifier = ({ language }) => {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Camera state
  const [showCamera, setShowCamera] = useState(false);
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const predict = async () => {
    if (!image) return alert("Please upload an image");
    
    setLoading(true);
    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch(
        `/predict?lang=${language}`,
        { method: "POST", body: formData }
      );
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Failed to connect to AI server. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  // Camera Functions
  const startCamera = async () => {
    setShowCamera(true);
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" }
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert("Camera access denied or not available. Please use Gallery instead.");
      setShowCamera(false);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
    setStream(null);
    setShowCamera(false);
  };

  const takePhoto = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob((blob) => {
      // Create a file object from blob
      const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      stopCamera();
    }, 'image/jpeg');
  };

  // Cleanup stream on unmount
  useEffect(() => {
    return () => {
      if (stream) stream.getTracks().forEach(t => t.stop());
    };
  }, [stream]);

  const isError = result && result.disease === "Invalid Image";
  const isHealthy = result && result.disease && result.disease.toLowerCase().includes('healthy');

  return (
    <section className="classifier-section">
      <div className="classifier-card">
        
        {preview ? (
           <div className="upload-area has-file" onClick={() => { setPreview(null); setImage(null); setResult(null); }}>
             <img src={preview} alt="preview" className="preview-img" style={{marginBottom: '10px'}} />
             <p style={{fontSize:'0.9rem', color:'var(--text-muted)'}}>
               ({language === 'hi' ? 'छवि हटाने के लिए क्लिक करें' : language === 'mr' ? 'प्रतिमा काढण्यासाठी क्लिक करा' : 'Click to remove image'})
             </p>
           </div>
        ) : (
          <div className="upload-area">
              <h3 style={{marginBottom:'10px'}}>
                {language === 'hi' ? 'चित्र अपलोड करें' : language === 'mr' ? 'फोटो अपलोड करा' : 'Upload Image'}
              </h3>
              <div style={{display:'flex', gap:'20px', width: '100%', justifyContent:'center'}}>
                {/* GALLERY UPLOAD */}
                <label className="action-label" style={{display:'flex', flexDirection:'column', alignItems:'center', cursor:'pointer', padding:'20px', background:'#f8fafc', borderRadius:'12px', border:'1px solid #e2e8f0', width:'45%'}}>
                  <input type="file" accept="image/*" onChange={handleImageChange} style={{display:'none'}} />
                  <UploadCloud size={32} color="var(--primary)" style={{marginBottom:'10px'}} />
                  <span style={{fontWeight:'600', fontSize:'0.95rem'}}>Gallery</span>
                </label>
                
                {/* CAMERA BUTTON (Triggers MediaDevices API) */}
                <div onClick={startCamera} className="action-label" style={{display:'flex', flexDirection:'column', alignItems:'center', cursor:'pointer', padding:'20px', background:'#f8fafc', borderRadius:'12px', border:'1px solid #e2e8f0', width:'45%'}}>
                  <Camera size={32} color="var(--primary)" style={{marginBottom:'10px'}} />
                  <span style={{fontWeight:'600', fontSize:'0.95rem'}}>Camera</span>
                </div>
              </div>
          </div>
        )}

        {/* Camera Modal overlay */}
        {showCamera && (
          <div style={{
            position: 'fixed', top: 0, left: 0, width: '100%', height: '100%',
            background: 'rgba(0,0,0,0.9)', zIndex: 9999, display: 'flex',
            flexDirection: 'column', justifyContent: 'center', alignItems: 'center'
          }}>
            <button onClick={stopCamera} style={{
              position: 'absolute', top: '20px', right: '20px', 
              background: 'white', borderRadius: '50%', padding: '10px', 
              border: 'none', cursor: 'pointer', zIndex: 10000,
              display: 'flex', justifyContent: 'center', alignItems: 'center'
            }}>
              <X size={24} color="black" />
            </button>
            
            <div style={{ width: '100%', maxWidth: '600px', padding: '20px', position: 'relative', zIndex: 9999, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <video 
                ref={videoRef} 
                autoPlay 
                playsInline 
                style={{width: '100%', borderRadius: '12px', backgroundColor: 'black', minHeight: '300px'}}
              />
              <button onClick={takePhoto} style={{
                marginTop: '30px', padding: '15px 40px', fontSize: '1.2rem',
                background: 'var(--primary)', color: 'white', borderRadius: '30px',
                border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '10px'
              }}>
                <Camera size={24} /> Capture Leaf
              </button>
            </div>
          </div>
        )}

        <button 
          className="btn-primary" 
          onClick={predict} 
          disabled={!image || loading}
          style={{marginTop: '20px'}}
        >
          {loading ? (
            <><Loader2 className="animate-spin" /> {language === 'hi' ? 'विश्लेषण...' : language === 'mr' ? 'विश्लेषण करत आहे...' : 'Analyzing...'}</>
          ) : (
             language === 'hi' ? 'रोग की भविष्यवाणी करें' : language === 'mr' ? 'रोगाचा अंदाज करा' : 'Predict Disease'
          )}
        </button>

        {result && (
          <div className={`result-box ${isError ? 'error' : ''}`}>
            <h3>
              {isError ? <AlertTriangle size={24} style={{verticalAlign: 'middle', marginRight: '8px'}}/> : 
               isHealthy ? <CheckCircle size={24} color="var(--primary)" style={{verticalAlign: 'middle', marginRight: '8px'}}/> : null}
              {result.message}
            </h3>
            


            
            {result.recommendations && result.recommendations.length > 0 && (
              <>
                <h4 style={{marginTop: '15px'}}>{language === 'hi' ? 'सुझाव' : language === 'mr' ? 'शिफारसी' : 'Recommendations'}:</h4>
                <ul>
                  {result.recommendations.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </div>
    </section>
  );
};

export default DiseaseClassifier;

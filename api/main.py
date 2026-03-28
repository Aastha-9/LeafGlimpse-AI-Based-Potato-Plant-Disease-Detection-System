from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

from translations import translations
from recommendations import recommendations
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables forcibly so hot-reloads detect changes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=env_path, override=True)

# Configuration
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY is missing!", flush=True)

genai.configure(api_key=api_key)

# Global model cache to avoid repeated API calls
MODEL_CACHE = {"chat": None, "vision": None}

def get_model(mode="chat"):
    """
    Dynamically finds a working model for the given mode.
    Mode: 'chat' (text-only) or 'vision' (multimodal)
    """
    if MODEL_CACHE[mode]:
        return MODEL_CACHE[mode]

    # Priority defaults
    defaults = {
        "chat": ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-pro"],
        "vision": ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-pro-vision"]
    }

    # 1. Try defaults first
    for model_name in defaults[mode]:
        try:
            m = genai.GenerativeModel(model_name)
            # Test it briefly
            m.generate_content("test", generation_config={"max_output_tokens": 1})
            MODEL_CACHE[mode] = m
            print(f"Successfully loaded '{model_name}' for {mode} mode.", flush=True)
            return m
        except Exception:
            continue

    # 2. If defaults fail, list all available models and find a compatible one
    try:
        print(f"Default models failed for {mode}. Listing available models...", flush=True)
        available = genai.list_models()
        for m_info in available:
            if 'generateContent' in m_info.supported_generation_methods:
                # Basic logic: vision models usually have 'vision' or 'flash' or '1.5' in name
                is_vision_capable = 'vision' in m_info.name or 'flash' in m_info.name or '1.5' in m_info.name
                
                if (mode == "vision" and is_vision_capable) or (mode == "chat"):
                    try:
                        m = genai.GenerativeModel(m_info.name)
                        m.generate_content("test", generation_config={"max_output_tokens": 1})
                        MODEL_CACHE[mode] = m
                        print(f"Dynamically discovered and loaded '{m_info.name}' for {mode} mode.", flush=True)
                        return m
                    except Exception:
                        continue
    except Exception as e:
        print(f"Critical error during model discovery: {e}", flush=True)

    # Final fallback to first default even if it might fail (will show error in API)
    return genai.GenerativeModel(defaults[mode][0])

app = FastAPI()

# ========================
# CORS
# ========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Class names (same order as training)
# ========================
class_names = [
    "Potato___Late_blight",
    "Potato___Early_blight",
    "Potato___healthy"
]

# ========================
# Load latest SavedModel
# ========================
MODEL_DIR = "../models"
latest_version = max([int(i) for i in os.listdir(MODEL_DIR) if i.isdigit()])

model = tf.keras.layers.TFSMLayer(
    f"{MODEL_DIR}/{latest_version}",
    call_endpoint="serve"
)

# ========================
# Routes
# ========================
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, lang: str = Query("en")):
    lang_map = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
    target_lang = lang_map.get(lang, "English")
    try:
        # Get a working model dynamically
        model = get_model("chat")
        response = model.generate_content(
            f"You are a highly knowledgeable agricultural AI assistant. You MUST reply ONLY in the {target_lang} language! Answer the user's question clearly and concisely: {request.message}"
        )
        return {"reply": response.text}
    except Exception as e:
        error_msg = f"I'm sorry, I'm having trouble connecting to my agricultural brain right now. {e}"
        if lang == "hi":
            error_msg = f"मुझे खेद है, मुझे अभी अपने कृषि नेटवर्क से जुड़ने में समस्या हो रही है। {e}"
        elif lang == "mr":
            error_msg = f"क्षमस्व, मला आता माझ्या कृषी नेटवर्कशी कनेक्ट होण्यास अडचण येत आहे. {e}"
        return {"reply": error_msg}

@app.get("/blogs")
def get_blogs(lang: str = Query("en")):
    import random
    # Multilingual structured mock blogs
    db = {
        "en": [
            {"id": 1, "title": "Understanding Late Blight", "category": "Disease", "excerpt": "Late blight is a devastating potato disease. Learn how to identify its early signs.", "content": "Late blight is caused by the oomycete Phytophthora infestans. It thrives in cool, humid weather and can destroy an entire potato crop within days. Early symptoms include water-soaked spots on leaves that turn brown or black, often with a white fungal growth on the underside. Preventive measures include planting certified disease-free seed potatoes, ensuring good spacing, and applying protective fungicides."},
            {"id": 2, "title": "Top 5 Fungicides", "category": "Remedies", "excerpt": "A deep dive into Mancozeb and Metalaxyl for agricultural use.", "content": "When fighting potato blight, timely application of fungicides is crucial. Mancozeb is an excellent protectant fungicide that forms a protective barrier on the leaf. Metalaxyl offers systemic protection, meaning it gets absorbed into the plant tissue. For best results, rotate between different classes of fungicides to prevent the disease from building resistance."},
            {"id": 3, "title": "Optimizing Crop Yield", "category": "Crops", "excerpt": "How proper drainage and soil management can double your potato harvest.", "content": "Potatoes require well-drained, loose soil to grow unobstructed. Soil compaction can severely restrict tuber expansion and lead to deformed potatoes. Raised beds or hilling up soil around the base of the plant can improve drainage and protect shallow tubers from sun exposure, which turns them green and toxic. Regular crop rotation is also vital."},
            {"id": 4, "title": "Identifying Early Blight", "category": "Disease", "excerpt": "Spotting early blight before it ruins your foliage.", "content": "Early blight, caused by Alternaria solani, primarily affects older leaves first. You'll notice small, dark, circular spots that often have concentric rings, like a target. Warm, humid weather accelerates its spread. To prevent early blight, ensure adequate fertilization (especially nitrogen) and avoid overhead irrigation which keeps leaves wet unnecessarily."},
            {"id": 5, "title": "Organic Potato Farming", "category": "Crops", "excerpt": "Grow healthy crops without synthetic chemicals.", "content": "Organic potato farming relies on building healthy soil biology using compost, cover crops, and organic amendments. Pest management is done through natural predators, crop rotation, and botanical sprays like neem oil. While it requires more intensive labor and planning, organic potatoes often command a premium price in local markets."},
            {"id": 6, "title": "Watering Strategies", "category": "Remedies", "excerpt": "When, how, and exactly how much to water your potato plants.", "content": "Potatoes need consistent moisture, especially during tuber formation. Inconsistent watering leads to cracking and malformed potatoes. The general rule is 1 to 2 inches of water per week. Drip irrigation is highly recommended over sprinklers because it delivers water directly to the root zone and keeps the leaves dry, dramatically lowering the risk of fungal diseases like blight."},
        ],
        "hi": [
            {"id": 1, "title": "लेट ब्लाइट को समझना", "category": "रोग", "excerpt": "लेट ब्लाइट एक विनाशकारी आलू की बीमारी है। इसके शुरुआती लक्षणों को पहचानना सीखें।", "content": "लेट ब्लाइट फाइटोफ्थोरा इन्फेस्टैन्स के कारण होता है। यह ठंडे और नम मौसम में फलता-फूलता है। शुरुआती लक्षणों में पत्तियों पर पानी से भीगे धब्बे शामिल हैं जो भूरे या काले हो जाते हैं। निवारक उपायों में प्रमाणित रोग-मुक्त बीज आलू लगाना और सुरक्षात्मक कवकनाशी लागू करना शामिल है।"},
            {"id": 2, "title": "शीर्ष 5 फफूंदनाशक", "category": "उपाय", "excerpt": "कृषि उपयोग के लिए मैंकोजेब और मेटलैक्सिल पर एक गहरा अध्ययन।", "content": "आलू के ब्लाइट से लड़ते समय, कवकनाशी का समय पर उपयोग महत्वपूर्ण है। मैंकोजेब एक बेहतरीन रक्षक कवकनाशी है जो पत्ती पर एक सुरक्षात्मक परत बनाता है। मेटलैक्सिल प्रणालीगत सुरक्षा प्रदान करता है। सर्वोत्तम परिणामों के लिए, विभिन्न कवकनाशी का उपयोग बदल-बदल कर करें।"},
            {"id": 3, "title": "फसल की उपज को अनुकूलित करना", "category": "फसलें", "excerpt": "उचित जल निकासी और मिट्टी प्रबंधन आपके आलू की फसल को कैसे दोगुना कर सकता है।", "content": "आलू को बिना रुकावट बढ़ने के लिए अच्छी जल निकासी वाली हल्की मिट्टी की आवश्यकता होती है। मिट्टी का सख्त होना कंदों के विकास को रोक सकता है। पौधों के आधार पर मिट्टी चढ़ाने से जल निकासी में सुधार होता है और कंदों को सूरज की रोशनी से बचाया जा सकता है।"},
            {"id": 4, "title": "अर्ली ब्लाइट की पहचान", "category": "रोग", "excerpt": "पत्ते खराब होने से पहले अर्ली ब्लाइट का पता लगाएं।", "content": "ऑल्टरनेरिया सोलानी के कारण होने वाला अर्ली ब्लाइट मुख्य रूप से पुरानी पत्तियों को प्रभावित करता है। आपको छोटे, गहरे रंग के गोल धब्बे दिखाई देंगे जिनमें अक्सर लक्ष्य की तरह गाढ़ा छल्ला होता है। गर्म और नम मौसम इसके प्रसार को तेज करता है।"},
            {"id": 5, "title": "जैविक आलू की खेती", "category": "फसलें", "excerpt": "बिना रसायनों के स्वस्थ फसलें उगाएं।", "content": "जैविक आलू की खेती खाद और फसल चक्रण का उपयोग करके स्वस्थ मिट्टी बनाने पर निर्भर करती है। नीम के तेल जैसे प्राकृतिक स्प्रे का उपयोग कीट प्रबंधन के लिए किया जाता है। हालांकि इसमें अधिक श्रम लगता है, जैविक आलू अक्सर बाजार में ज्यादा कीमत देते हैं।"},
            {"id": 6, "title": "सिंचाई की रणनीतियाँ", "category": "उपाय", "excerpt": "आलू के पौधों को कब और कितना पानी देना है।", "content": "आलू को लगातार नमी की आवश्यकता होती है, खासकर कंद बनने के दौरान। ड्रिप सिंचाई की अत्यधिक अनुशंसा की जाती है क्योंकि यह पानी को सीधे जड़ों तक पहुंचाता है और पत्तियों को सूखा रखता है, जिससे फंगल रोगों का जोखिम कम हो जाता है।"},
        ],
        "mr": [
            {"id": 1, "title": "लेट ब्लाइट समजून घेणे", "category": "रोग", "excerpt": "लेट ब्लाइट हा बटाट्यावरील एक भयंकर रोग आहे. त्याची सुरुवातीची लक्षणे कशी ओळखावीत ते शिका.", "content": "लेट ब्लाइट फायटोप्थोरा इन्फेस्टन्समुळे होतो. हा रोग थंड आणि दमट हवामानात वेगाने पसरतो. सुरुवातीच्या लक्षणांमध्ये पानांवर पाण्यासारखे डाग दिसतात जे नंतर काळे पडतात. प्रतिबंधात्मक उपायांमध्ये प्रमाणित रोगमुक्त बियाणे वापरणे आणि बुरशीनाशकांची फवारणी करणे समाविष्ट आहे."},
            {"id": 2, "title": "नवीन 5 बुरशीनाशके", "category": "उपाय", "excerpt": "शेतीसाठी मॅनकोझेब आणि मेटलॅक्सिलचा सखोल अभ्यास.", "content": "बटाट्यावरील रोगांशी लढताना, बुरशीनाशकांचा वेळेवर वापर अत्यंत महत्त्वाचा आहे. मॅनकोझेब हे पानांवर एक संरक्षक आवरण तयार करते, तर मेटलॅक्सिल प्रणालीगत संरक्षण देते. चांगल्या परिणामांसाठी विविध प्रकारच्या बुरशीनाशकांचा आलटून पालटून वापर करा."},
            {"id": 3, "title": "पिकांचे उत्पन्न वाढवणे", "category": "पिके", "excerpt": "पाण्याचा योग्य निचरा आणि माती व्यवस्थापन तुमचे बटाट्याचे पीक कसे दुप्पट करू शकते.", "content": "बटाट्याच्या चांगल्या वाढीसाठी पाण्याचा उत्तम निचरा होणारी भुसभुशीत माती आवश्यक असते. माती घट्ट असल्यास कंदांची वाढ खुंटते. बटाट्याच्या रोपाच्या बुंध्याजवळ मातीची भर दिल्यास पाण्याचा निचरा चांगला होतो आणि कंदांचे सूर्यप्रकाशापासून संरक्षण होते."},
            {"id": 4, "title": "अर्ली ब्लाइटची ओळख", "category": "रोग", "excerpt": "पाने खराब होण्यापूर्वी अर्ली ब्लाइट कसा ओळखावा.", "content": "अल्टरनेरिया सोलानीमुळे होणारा अर्ली ब्लाइट प्रामुख्याने जुन्या पानांवर परिणाम करतो. तुम्हाला पानांवर लहान, गडद, गोलाकार डाग दिसतील. उष्ण आणि दमट हवामानात या रोगाचा प्रसार वेगाने होतो. अर्ली ब्लाइट टाळण्यासाठी पिकांना योग्य खते द्या."},
            {"id": 5, "title": "सेंद्रिय बटाटा शेती", "category": "पिके", "excerpt": "कोणत्याही रसायनांशिवाय निरोगी पिके घ्या.", "content": "सेंद्रिय बटाटा शेती नैसर्गिक खत आणि योग्य पीक व्यवस्थापनावर अवलंबून असते. कीड व्यवस्थापनासाठी कडुनिंबाच्या तेलासारख्या नैसर्गिक फवारण्यांचा वापर करा. यात जास्त मेहनत लागत असली तरी सेंद्रिय बटाट्यांना बाजारात जास्त भाव मिळतो."},
            {"id": 6, "title": "सिंचन पद्धती", "category": "उपाय", "excerpt": "बटाट्याच्या झाडांना पाणी किती आणि कसे द्यावे.", "content": "बटाट्यांना वाढीच्या काळात सतत ओलावा लागतो. ठिबक सिंचनाची शिफारस केली जाते कारण यामुळे बुरशीजन्य रोगांचा धोका कमी होतो. तुषार सिंचनामुळे पाने ओली राहून रोगांचा प्रादुर्भाव होऊ शकतो."},
        ]
    }
    
    active_blogs = db.get(lang, db["en"])
    return random.sample(active_blogs, min(6, len(active_blogs)))

# ========================
# Serve React Frontend
# ========================
BUILD_DIR = os.path.join(BASE_DIR, "..", "frontend", "build")

if os.path.exists(BUILD_DIR):
    # Serve the static JS/CSS/media assets
    app.mount("/static", StaticFiles(directory=os.path.join(BUILD_DIR, "static")), name="static")

    @app.get("/")
    def serve_react_root():
        return FileResponse(os.path.join(BUILD_DIR, "index.html"))

    @app.get("/{full_path:path}")
    def serve_react_spa(full_path: str):
        """Catch-all: send all non-API requests to the React index.html."""
        file_path = os.path.join(BUILD_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(BUILD_DIR, "index.html"))
else:
    @app.get("/")
    def home():
        return {"message": "Potato Disease API is running! Build the React frontend to serve the UI."}

@app.get("/debug")
def debug():
    img_array = np.ones((1, 256, 256, 3))
    predictions = model(img_array)
    return {
        "predictions_type": str(type(predictions)),
        "predictions_repr": str(predictions)
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lang: str = Query("en")   # en | hi | mr
):
    import asyncio
    
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Resize for Gemini to speed up upload
    gemini_image = image.resize((256, 256))
    
    # Prepare local model array (no manual /255 — model handles rescaling internally)
    img_array = np.array(image.resize((256, 256)), dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # ---- TEST-TIME AUGMENTATION (TTA) ----
    # Run through 5 augmented versions and average to improve real-world accuracy
    def augment(arr):
        """Returns a list of augmented numpy arrays from a single (1,H,W,3) array."""
        variants = [arr]
        # horizontal flip
        variants.append(arr[:, :, ::-1, :])
        # vertical flip
        variants.append(arr[:, ::-1, :, :])
        # slight brightness boost (+10%)
        variants.append(np.clip(arr * 1.1, 0, 255))
        # slight brightness dim (-10%)
        variants.append(np.clip(arr * 0.9, 0, 255))
        return variants
    
    tta_variants = augment(img_array)

    # ---- LOCAL PRE-FILTER: reject obvious non-leaf images ----
    img_np = np.array(image.resize((128, 128)), dtype=np.float32)
    
    R = img_np[:, :, 0]
    G = img_np[:, :, 1]
    B = img_np[:, :, 2]
    
    # Check: Is there ANY meaningful color variation? 
    # Only rejects truly monochrome images (pure drawings/sketches)
    # Real leaves — even diseased/dark ones — have some saturation
    max_channel = np.maximum(np.maximum(R, G), B)
    min_channel = np.minimum(np.minimum(R, G), B)
    saturation_map = max_channel - min_channel
    avg_saturation = np.mean(saturation_map)
    
    # Very lenient: only reject if almost completely grayscale (saturation < 8)
    # A pure black & white pencil drawing scores ~0-5, real photos score 20-80+
    is_colorless = avg_saturation < 8.0
    
    if is_colorless:
        return {
            "disease": "Invalid Image",
            "confidence": 0.0,
            "message": "The image appears to be a black & white drawing or sketch, not a real plant leaf.",
            "recommendations": [
                "Ensure the photo is a real photograph (not a drawing or artwork).",
                "Ensure the leaf is clearly visible.",
                "Ensure the plant is a potato crop."
            ]
        }
    # ----------------------------------------------------------

    # 1. Define the cloud API Task
    async def validate_image():
        try:
            prompt = (
                "Look at this low-res image. "
                "Is this a picture that prominently features a potato leaf? "
                "If it IS a potato leaf, reply ONLY with the exact word 'POTATO'. "
                "If it is a leaf of a different plant, reply with the exact plant name (e.g., 'Tomato leaf', 'Oak leaf'). "
                "If it is not a leaf at all, reply 'NOT A LEAF'."
            )
            # 1. Get a working vision model dynamically
            model_engine = get_model("vision")
            resp = await asyncio.to_thread(
                model_engine.generate_content,
                [prompt, gemini_image],
                generation_config={"max_output_tokens": 10}
            )
            return resp.text.strip().upper()
        except Exception as e:
            # If Gemini is unavailable, skip validation and trust the local model
            print(f"Gemini API unavailable (skipping validation): {e}", flush=True)
            return "POTATO"
            
    # 2. Define the local TF task (with TTA averaging)
    def run_local():
        all_preds = []
        for variant in tta_variants:
            preds = model(variant)
            if isinstance(preds, dict):
                preds = list(preds.values())[0]
            all_preds.append(preds.numpy()[0])
        # Average across all augmented versions for more robust prediction
        return np.mean(all_preds, axis=0)

    # Run BOTH at the EXACT same time!
    verdict_task = asyncio.create_task(validate_image())
    local_task = asyncio.to_thread(run_local)
    
    verdict, pred_vals = await asyncio.gather(verdict_task, local_task)
    
    # Only reject if Gemini explicitly identified it as non-potato (not on API errors)
    if verdict != "POTATO" and not verdict.startswith("API_ERROR"):
        friendly_message = f"This appears to be: {verdict.title()}. Please upload a clear photo of a Potato leaf."
        if "NOT A LEAF" in verdict:
            friendly_message = "This image does not appear to be a leaf. Please upload a clear photo of a Potato leaf."
            
        return {
            "disease": "Invalid Image",
            "confidence": 0.0,
            "message": friendly_message,
            "recommendations": ["Ensure the photo is strictly of a single plant leaf.", "Ensure the plant is a potato crop."]
        }

    # Use the local model prediction
    index = int(np.argmax(pred_vals))
    confidence = float(np.max(pred_vals)) * 100
    predicted_class = class_names[index]
    
    # DEBUG: Log raw softmax probabilities to diagnose model calibration
    print(f"\n--- PREDICTION DEBUG ---", flush=True)
    for i, cls in enumerate(class_names):
        print(f"  {cls}: {pred_vals[i]*100:.4f}%", flush=True)
    print(f"  → Winner: {predicted_class} ({confidence:.4f}%)", flush=True)
    print(f"------------------------\n", flush=True)

    return {
        "disease": predicted_class,
        "confidence": round(confidence, 2),
        "message": translations[predicted_class].get(lang, translations[predicted_class]["en"]),
        "recommendations": recommendations[predicted_class].get(lang, recommendations[predicted_class]["en"])
    }

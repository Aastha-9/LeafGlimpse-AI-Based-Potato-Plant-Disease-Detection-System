import asyncio
import io
import os
import random

import google.generativeai as genai
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from recommendations import recommendations
from translations import translations

# ---------------------------------------------------------------------------
# Environment & API key
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"), override=True)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY is missing!", flush=True)

genai.configure(api_key=api_key)

# ---------------------------------------------------------------------------
# Gemini model cache  (avoids repeated discovery on every request)
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {"chat": None, "vision": None}

_DEFAULTS = {
    "chat":   ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash",
               "gemini-1.5-flash-latest", "gemini-1.5-pro", "gemini-pro"],
    "vision": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash",
               "gemini-1.5-flash-latest", "gemini-1.5-pro", "gemini-pro-vision"],
}


def get_model(mode: str = "chat"):
    """
    Dynamically finds and caches a working Gemini model.
    mode: 'chat' (text-only) or 'vision' (multimodal)
    """
    if _MODEL_CACHE[mode]:
        return _MODEL_CACHE[mode]

    # 1. Try priority defaults first
    for model_name in _DEFAULTS[mode]:
        try:
            m = genai.GenerativeModel(model_name)
            if mode == "vision":
                dummy_img = Image.new("RGB", (10, 10))
                m.generate_content(
                    ["test", dummy_img],
                    generation_config={"max_output_tokens": 1},
                )
            else:
                m.generate_content("test", generation_config={"max_output_tokens": 1})
            _MODEL_CACHE[mode] = m
            print(f"Loaded '{model_name}' for {mode} mode.", flush=True)
            return m
        except Exception as e:
            print(f"Model '{model_name}' unavailable: {e}", flush=True)

    # 2. Fallback: discover all available models dynamically
    try:
        print(f"Defaults failed for {mode}. Scanning available models…", flush=True)
        for m_info in genai.list_models():
            if "generateContent" not in m_info.supported_generation_methods:
                continue
            is_vision = any(
                kw in m_info.name.lower()
                for kw in ["vision", "flash", "1.5", "2.0", "2.5"]
            )
            if mode == "vision" and not is_vision:
                continue
            try:
                m = genai.GenerativeModel(m_info.name)
                m.generate_content("test", generation_config={"max_output_tokens": 1})
                _MODEL_CACHE[mode] = m
                print(f"Discovered '{m_info.name}' for {mode} mode.", flush=True)
                return m
            except Exception:
                continue
    except Exception as e:
        print(f"Critical error during model discovery: {e}", flush=True)

    # 3. Last resort — return first default (may fail at call-time)
    return genai.GenerativeModel(_DEFAULTS[mode][0])


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="LeafGlimpse API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# TensorFlow model  (class names must match training order)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Potato___Late_blight",
    "Potato___Early_blight",
    "Potato___healthy",
]

MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
_latest_version = max(int(v) for v in os.listdir(MODEL_DIR) if v.isdigit())
tf_model = tf.keras.layers.TFSMLayer(
    os.path.join(MODEL_DIR, str(_latest_version)),
    call_endpoint="serve",
)

# ---------------------------------------------------------------------------
# Helper: safety settings for Gemini (lenient for agricultural content)
# ---------------------------------------------------------------------------
_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, lang: str = Query("en")):
    """Agricultural chatbot powered by Gemini."""
    lang_map = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
    target_lang = lang_map.get(lang, "English")
    try:
        gemini = get_model("chat")
        chat = gemini.start_chat(history=[])
        query = (
            f"You are a highly knowledgeable agricultural AI assistant. "
            f"You MUST reply ONLY in the {target_lang} language! "
            f"Answer the user's question clearly and concisely: {request.message}"
        )
        response = chat.send_message(query)
        return {"reply": response.text}
    except Exception as e:
        error_msg = str(e)
        print(f"Chatbot Error: {error_msg}", flush=True)
        if "429" in error_msg or "quota" in error_msg.lower():
            quota_msgs = {
                "en": (
                    "I've answered so many questions today that I need a quick rest! "
                    "Please try again in a few minutes."
                ),
                "hi": (
                    "मैंने आज इतने सारे सवालों के जवाब दिए हैं कि मुझे थोड़ी देर "
                    "आराम की ज़रूरत है! कृपया कुछ मिनटों बाद फिर से प्रयास करें।"
                ),
                "mr": (
                    "मी आज इतक्या प्रश्नांची उत्तरे दिली आहेत की मला थोड्या "
                    "विश्रांतीची गरज आहे! कृपया काही मिनिटांनंतर पुन्हा प्रयत्न करा."
                ),
            }
            return {"reply": quota_msgs.get(lang, quota_msgs["en"])}
        return {
            "reply": (
                "I'm sorry, I'm having trouble connecting to my agricultural brain "
                "right now. Please try again later."
            )
        }


@app.get("/blogs")
def get_blogs(lang: str = Query("en")):
    """Returns 6 random agricultural blog posts in the requested language."""
    db = {
        "en": [
            {
                "id": 1, "title": "Understanding Late Blight", "category": "Disease",
                "excerpt": "Late blight is a devastating potato disease. Learn how to identify its early signs.",
                "content": (
                    "Late blight is caused by the oomycete Phytophthora infestans. It thrives in cool, "
                    "humid weather and can destroy an entire potato crop within days. Early symptoms include "
                    "water-soaked spots on leaves that turn brown or black, often with a white fungal growth "
                    "on the underside. Preventive measures include planting certified disease-free seed "
                    "potatoes, ensuring good spacing, and applying protective fungicides."
                ),
            },
            {
                "id": 2, "title": "Top 5 Fungicides", "category": "Remedies",
                "excerpt": "A deep dive into Mancozeb and Metalaxyl for agricultural use.",
                "content": (
                    "When fighting potato blight, timely application of fungicides is crucial. Mancozeb is "
                    "an excellent protectant fungicide that forms a protective barrier on the leaf. Metalaxyl "
                    "offers systemic protection, meaning it gets absorbed into the plant tissue. For best "
                    "results, rotate between different classes of fungicides to prevent resistance build-up."
                ),
            },
            {
                "id": 3, "title": "Optimizing Crop Yield", "category": "Crops",
                "excerpt": "How proper drainage and soil management can double your potato harvest.",
                "content": (
                    "Potatoes require well-drained, loose soil to grow unobstructed. Soil compaction can "
                    "severely restrict tuber expansion. Raised beds or hilling up soil around the plant base "
                    "improves drainage and protects tubers from sun exposure. Regular crop rotation is vital."
                ),
            },
            {
                "id": 4, "title": "Identifying Early Blight", "category": "Disease",
                "excerpt": "Spotting early blight before it ruins your foliage.",
                "content": (
                    "Early blight, caused by Alternaria solani, primarily affects older leaves first. You'll "
                    "notice small, dark, circular spots with concentric rings, like a target. Warm, humid "
                    "weather accelerates its spread. Ensure adequate fertilization (especially nitrogen) and "
                    "avoid overhead irrigation."
                ),
            },
            {
                "id": 5, "title": "Organic Potato Farming", "category": "Crops",
                "excerpt": "Grow healthy crops without synthetic chemicals.",
                "content": (
                    "Organic potato farming relies on building healthy soil biology using compost, cover "
                    "crops, and organic amendments. Pest management uses natural predators, crop rotation, "
                    "and botanical sprays like neem oil. Organic potatoes often command a premium price."
                ),
            },
            {
                "id": 6, "title": "Watering Strategies", "category": "Remedies",
                "excerpt": "When, how, and exactly how much to water your potato plants.",
                "content": (
                    "Potatoes need consistent moisture, especially during tuber formation — about 1–2 inches "
                    "per week. Drip irrigation is recommended over sprinklers because it keeps leaves dry, "
                    "dramatically lowering the risk of fungal diseases like blight."
                ),
            },
        ],
        "hi": [
            {
                "id": 1, "title": "लेट ब्लाइट को समझना", "category": "रोग",
                "excerpt": "लेट ब्लाइट एक विनाशकारी आलू की बीमारी है। इसके शुरुआती लक्षणों को पहचानना सीखें।",
                "content": (
                    "लेट ब्लाइट फाइटोफ्थोरा इन्फेस्टैन्स के कारण होता है। यह ठंडे और नम मौसम में "
                    "फलता-फूलता है। निवारक उपायों में प्रमाणित रोग-मुक्त बीज आलू लगाना और सुरक्षात्मक "
                    "कवकनाशी लागू करना शामिल है।"
                ),
            },
            {
                "id": 2, "title": "शीर्ष 5 फफूंदनाशक", "category": "उपाय",
                "excerpt": "कृषि उपयोग के लिए मैंकोजेब और मेटलैक्सिल पर एक गहरा अध्ययन।",
                "content": (
                    "आलू के ब्लाइट से लड़ते समय, कवकनाशी का समय पर उपयोग महत्वपूर्ण है। मैंकोजेब पत्ती पर "
                    "एक सुरक्षात्मक परत बनाता है। मेटलैक्सिल प्रणालीगत सुरक्षा प्रदान करता है।"
                ),
            },
            {
                "id": 3, "title": "फसल की उपज को अनुकूलित करना", "category": "फसलें",
                "excerpt": "उचित जल निकासी और मिट्टी प्रबंधन आपके आलू की फसल को कैसे दोगुना कर सकता है।",
                "content": (
                    "आलू को बिना रुकावट बढ़ने के लिए अच्छी जल निकासी वाली हल्की मिट्टी की आवश्यकता होती है। "
                    "मिट्टी का सख्त होना कंदों के विकास को रोक सकता है।"
                ),
            },
            {
                "id": 4, "title": "अर्ली ब्लाइट की पहचान", "category": "रोग",
                "excerpt": "पत्ते खराब होने से पहले अर्ली ब्लाइट का पता लगाएं।",
                "content": (
                    "ऑल्टरनेरिया सोलानी के कारण होने वाला अर्ली ब्लाइट मुख्य रूप से पुरानी पत्तियों को "
                    "प्रभावित करता है। गर्म और नम मौसम इसके प्रसार को तेज करता है।"
                ),
            },
            {
                "id": 5, "title": "जैविक आलू की खेती", "category": "फसलें",
                "excerpt": "बिना रसायनों के स्वस्थ फसलें उगाएं।",
                "content": (
                    "जैविक आलू की खेती खाद और फसल चक्रण का उपयोग करके स्वस्थ मिट्टी बनाने पर निर्भर करती है। "
                    "जैविक आलू अक्सर बाजार में ज्यादा कीमत देते हैं।"
                ),
            },
            {
                "id": 6, "title": "सिंचाई की रणनीतियाँ", "category": "उपाय",
                "excerpt": "आलू के पौधों को कब और कितना पानी देना है।",
                "content": (
                    "आलू को लगातार नमी की आवश्यकता होती है। ड्रिप सिंचाई की अत्यधिक अनुशंसा की जाती है "
                    "क्योंकि यह पत्तियों को सूखा रखता है, जिससे फंगल रोगों का जोखिम कम हो जाता है।"
                ),
            },
        ],
        "mr": [
            {
                "id": 1, "title": "लेट ब्लाइट समजून घेणे", "category": "रोग",
                "excerpt": "लेट ब्लाइट हा बटाट्यावरील एक भयंकर रोग आहे. त्याची सुरुवातीची लक्षणे कशी ओळखावीत ते शिका.",
                "content": (
                    "लेट ब्लाइट फायटोप्थोरा इन्फेस्टन्समुळे होतो. प्रतिबंधात्मक उपायांमध्ये प्रमाणित "
                    "रोगमुक्त बियाणे वापरणे आणि बुरशीनाशकांची फवारणी करणे समाविष्ट आहे."
                ),
            },
            {
                "id": 2, "title": "नवीन 5 बुरशीनाशके", "category": "उपाय",
                "excerpt": "शेतीसाठी मॅनकोझेब आणि मेटलॅक्सिलचा सखोल अभ्यास.",
                "content": (
                    "मॅनकोझेब हे पानांवर एक संरक्षक आवरण तयार करते, तर मेटलॅक्सिल प्रणालीगत संरक्षण देते. "
                    "चांगल्या परिणामांसाठी विविध बुरशीनाशकांचा आलटून पालटून वापर करा."
                ),
            },
            {
                "id": 3, "title": "पिकांचे उत्पन्न वाढवणे", "category": "पिके",
                "excerpt": "पाण्याचा योग्य निचरा आणि माती व्यवस्थापन तुमचे बटाट्याचे पीक कसे दुप्पट करू शकते.",
                "content": (
                    "बटाट्याच्या चांगल्या वाढीसाठी पाण्याचा उत्तम निचरा होणारी भुसभुशीत माती आवश्यक असते. "
                    "माती घट्ट असल्यास कंदांची वाढ खुंटते."
                ),
            },
            {
                "id": 4, "title": "अर्ली ब्लाइटची ओळख", "category": "रोग",
                "excerpt": "पाने खराब होण्यापूर्वी अर्ली ब्लाइट कसा ओळखावा.",
                "content": (
                    "अल्टरनेरिया सोलानीमुळे होणारा अर्ली ब्लाइट प्रामुख्याने जुन्या पानांवर परिणाम करतो. "
                    "उष्ण आणि दमट हवामानात या रोगाचा प्रसार वेगाने होतो."
                ),
            },
            {
                "id": 5, "title": "सेंद्रिय बटाटा शेती", "category": "पिके",
                "excerpt": "कोणत्याही रसायनांशिवाय निरोगी पिके घ्या.",
                "content": (
                    "सेंद्रिय बटाटा शेती नैसर्गिक खत आणि योग्य पीक व्यवस्थापनावर अवलंबून असते. "
                    "सेंद्रिय बटाट्यांना बाजारात जास्त भाव मिळतो."
                ),
            },
            {
                "id": 6, "title": "सिंचन पद्धती", "category": "उपाय",
                "excerpt": "बटाट्याच्या झाडांना पाणी किती आणि कसे द्यावे.",
                "content": (
                    "बटाट्यांना वाढीच्या काळात सतत ओलावा लागतो. ठिबक सिंचनाची शिफारस केली जाते कारण "
                    "यामुळे बुरशीजन्य रोगांचा धोका कमी होतो."
                ),
            },
        ],
    }
    active_blogs = db.get(lang, db["en"])
    return random.sample(active_blogs, min(6, len(active_blogs)))


# ---------------------------------------------------------------------------
# Serve React frontend (production build)
# ---------------------------------------------------------------------------
BUILD_DIR = os.path.join(BASE_DIR, "..", "frontend", "build")

if os.path.exists(BUILD_DIR):
    app.mount(
        "/static",
        StaticFiles(directory=os.path.join(BUILD_DIR, "static")),
        name="static",
    )

    @app.get("/")
    def serve_react_root():
        return FileResponse(os.path.join(BUILD_DIR, "index.html"))

    @app.get("/{full_path:path}")
    def serve_react_spa(full_path: str):
        """Catch-all: route all non-API paths to the React SPA."""
        file_path = os.path.join(BUILD_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(BUILD_DIR, "index.html"))

else:
    @app.get("/")
    def home():
        return {"message": "LeafGlimpse API is running. Build the React frontend to serve the UI."}


# ---------------------------------------------------------------------------
# Debug endpoint
# ---------------------------------------------------------------------------
@app.get("/debug")
def debug():
    img_array = np.ones((1, 256, 256, 3))
    predictions = tf_model(img_array)
    return {
        "predictions_type": str(type(predictions)),
        "predictions_repr": str(predictions),
    }


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lang: str = Query("en"),   # en | hi | mr
):
    """
    Classifies an uploaded potato leaf image.
    Runs Gemini vision validation and local TF model in parallel.
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize for Gemini (smaller = faster upload)
    gemini_image = image.resize((256, 256))

    # Prepare array for local TF model
    img_array = np.expand_dims(
        np.array(image.resize((256, 256)), dtype=np.float32), axis=0
    )

    # ------------------------------------------------------------------
    # Test-Time Augmentation (TTA) — average 5 variants for robustness
    # ------------------------------------------------------------------
    def _augment(arr: np.ndarray) -> list:
        return [
            arr,
            arr[:, :, ::-1, :],                  # horizontal flip
            arr[:, ::-1, :, :],                  # vertical flip
            np.clip(arr * 1.1, 0, 255),          # brightness +10 %
            np.clip(arr * 0.9, 0, 255),          # brightness -10 %
        ]

    tta_variants = _augment(img_array)

    # ------------------------------------------------------------------
    # Local pre-filter: reject obvious non-leaf images immediately
    # ------------------------------------------------------------------
    img_np = np.array(image.resize((128, 128)), dtype=np.float32)
    r_ch = img_np[:, :, 0].astype(float)
    g_ch = img_np[:, :, 1].astype(float)
    b_ch = img_np[:, :, 2].astype(float)

    # Colour saturation (HSV-style)
    max_c = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    min_c = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    avg_saturation = float(np.mean(max_c - min_c))

    # Excess Green Index — real leaves score ≥ 0, non-plants score negative
    avg_exg = float(np.mean(2.0 * g_ch - r_ch - b_ch))

    # White/document background
    white_ratio = float(np.mean(np.all(img_np > 200, axis=2)))
    color_std = float(np.std(img_np))

    is_colorless        = avg_saturation < 8.0
    is_document         = white_ratio > 0.85 or color_std < 10.0
    is_not_green_strict = avg_exg < -10.0 and not is_colorless

    print(
        f"PRE-FILTER | exg={avg_exg:.1f} sat={avg_saturation:.1f} "
        f"white={white_ratio:.2f} std={color_std:.1f}",
        flush=True,
    )

    if is_colorless or is_document or is_not_green_strict:
        if lang == "hi":
            if is_colorless:
                msg = "यह छवि एक श्वेत-श्याम रेखाचित्र या स्केच प्रतीत होती है, न कि असली पौधे की पत्ती।"
            elif is_document:
                msg = "यह छवि एक दस्तावेज़ या सफ़ेद पृष्ठभूमि वाला आरेख प्रतीत होती है, न कि असली पौधे की पत्ती।"
            else:
                msg = "छवि के रंग एक सामान्य पौधे की पत्ती से मेल नहीं खाते। कृपया हरी पत्ती का स्पष्ट फोटो अपलोड करें।"
            recoms = [
                "सुनिश्चित करें कि फोटो पौधे की पत्ती का वास्तविक छायाचित्र है।",
                "आरेख, फ्लोचार्ट या टेक्स्ट के स्क्रीनशॉट अपलोड करने से बचें।",
                "सुनिश्चित करें कि पौधा आलू की फसल है।",
            ]
        elif lang == "mr":
            if is_colorless:
                msg = "ही प्रतिमा एक कृष्णधवल रेखाचित्र किंवा स्केच वाटते, खरी वनस्पतीची पाने नाही."
            elif is_document:
                msg = "ही प्रतिमा पांढऱ्या पार्श्वभूमीचा दस्तऐवज किंवा आकृती वाटते, खरी वनस्पतीची पाने नाही."
            else:
                msg = "प्रतिमेचे रंग सामान्य वनस्पतींच्या पानांशी जुळत नाहीत. कृपया हिरव्या पानाचा स्पष्ट फोटो अपलोड करा."
            recoms = [
                "फोटो वनस्पतींच्या पानाचे प्रत्यक्ष छायाचित्र असल्याचे सुनिश्चित करा.",
                "आकृती, फ्लोचार्ट किंवा मजकुराचे स्क्रीनशॉट अपलोड करणे टाळा.",
                "वनस्पती बटाट्याचे पीक असल्याची खात्री करा.",
            ]
        else:
            if is_colorless:
                msg = "The image appears to be a black & white drawing or sketch, not a real plant leaf."
            elif is_document:
                msg = "The image appears to be a document or diagram, not a real plant leaf."
            else:
                msg = "The image colors do not match a typical plant leaf. Please upload a clear photo of a green leaf."
            recoms = [
                "Ensure the photo is a real photograph of a plant leaf.",
                "Avoid uploading diagrams, flowcharts, or screenshots of text.",
                "Ensure the plant is a potato crop.",
            ]
        return {"disease": "Invalid Image", "confidence": 0.0,
                "message": msg, "recommendations": recoms}

    # ------------------------------------------------------------------
    # Run Gemini vision validation + local TF model in parallel
    # ------------------------------------------------------------------
    async def _validate_with_gemini() -> str:
        try:
            prompt = (
                "Analyze this image. "
                "1. Is this a real potato plant leaf? (If not, reply 'NOT A LEAF'). "
                "2. If it IS a potato plant leaf, identify if it has 'Early Blight' "
                "(small concentric spots), 'Late Blight' (large dark brown/black "
                "water-soaked lesions), or is 'Healthy'. "
                "Reply strictly in the format: "
                "VERDICT: [POTATO/NOT A LEAF] | DISEASE: [Early Blight/Late Blight/Healthy/Unknown]\n"
                "Do NOT include any extra text, markdown, or explanations."
            )
            engine = get_model("vision")
            resp = await asyncio.to_thread(
                engine.generate_content,
                [prompt, gemini_image],
                generation_config={"max_output_tokens": 150},
                safety_settings=_SAFETY_SETTINGS,
            )
            if not resp.candidates or (
                resp.candidates[0].finish_reason.name not in ["STOP", "MAX_TOKENS"]
                and resp.candidates[0].finish_reason not in [1, 2]
            ):
                print(
                    f"Gemini blocked. Reason: "
                    f"{resp.candidates[0].finish_reason if resp.candidates else 'No candidates'}",
                    flush=True,
                )
                return "NOT A LEAF"
            return resp.text.strip().upper()
        except Exception as e:
            print(f"Gemini unavailable (stricter local filter active): {e}", flush=True)
            return "API_UNAVAILABLE"

    def _run_local() -> np.ndarray:
        all_preds = []
        for variant in tta_variants:
            preds = tf_model(variant)
            if isinstance(preds, dict):
                preds = list(preds.values())[0]
            all_preds.append(preds.numpy()[0])
        return np.mean(all_preds, axis=0)

    verdict, pred_vals = await asyncio.gather(
        asyncio.create_task(_validate_with_gemini()),
        asyncio.to_thread(_run_local),
    )

    # ------------------------------------------------------------------
    # Interpret Gemini verdict
    # ------------------------------------------------------------------
    gemini_unavailable = "API_UNAVAILABLE" in verdict
    is_potato = "POTATO" in verdict or gemini_unavailable

    gemini_disease = "UNKNOWN"
    if "DISEASE:" in verdict:
        gemini_disease = verdict.split("DISEASE:")[1].strip().split("|")[0].strip()

    # When Gemini is offline, tighten the ExG threshold as a safety net
    if gemini_unavailable and avg_exg < -2.0 and not is_colorless:
        print(f"Gemini offline — strict filter blocked image (exg={avg_exg:.1f})", flush=True)
        is_potato = False

    if not is_potato:
        if lang == "hi":
            friendly_msg = "यह छवि आलू की पत्ती जैसी नहीं लग रही है। कृपया आलू की पत्ती का स्पष्ट फोटो अपलोड करें।"
            recoms = [
                "सुनिश्चित करें कि फोटो केवल एक पौधे की पत्ती का है।",
                "सुनिश्चित करें कि पौधा आलू की फसल है।",
            ]
        elif lang == "mr":
            friendly_msg = "ही प्रतिमा बटाट्याचे पान वाटत नाही. कृपया बटाट्याच्या पानाचा स्पष्ट फोटो अपलोड करा."
            recoms = [
                "फोटो केवळ एका वनस्पतीच्या पानाचा असल्याची खात्री करा.",
                "वनस्पती बटाट्याचे पीक असल्याची खात्री करा.",
            ]
        else:
            friendly_msg = "This image does not appear to be a potato leaf. Please upload a clear photo of a Potato leaf."
            recoms = [
                "Ensure the photo is strictly of a single plant leaf.",
                "Ensure the plant is a potato crop.",
            ]
        return {"disease": "Invalid Image", "confidence": 0.0,
                "message": friendly_msg, "recommendations": recoms}

    # ------------------------------------------------------------------
    # Final prediction from local TF model
    # ------------------------------------------------------------------
    index = int(np.argmax(pred_vals))
    confidence = float(np.max(pred_vals)) * 100
    predicted_class = CLASS_NAMES[index]

    # Smart override: if Gemini sees Late Blight and local model is not fully certain
    if "LATE BLIGHT" in gemini_disease and confidence < 99.0:
        print(
            f"OVERRIDE: Gemini → Late Blight | Local → {predicted_class} ({confidence:.1f}%)",
            flush=True,
        )
        predicted_class = "Potato___Late_blight"
        confidence = 95.0

    print(
        f"RESULT: {predicted_class} ({confidence:.2f}%) | Gemini: {gemini_disease}",
        flush=True,
    )

    return {
        "disease": predicted_class,
        "confidence": round(confidence, 2),
        "message": translations[predicted_class].get(lang, translations[predicted_class]["en"]),
        "recommendations": recommendations[predicted_class].get(
            lang, recommendations[predicted_class]["en"]
        ),
    }


# ---------------------------------------------------------------------------
# Translation helper endpoint
# ---------------------------------------------------------------------------
@app.get("/translate_disease")
def translate_disease(disease: str, lang: str = Query("en")):
    """Returns translated message and recommendations for a given disease."""
    if disease == "Invalid Image":
        if lang == "hi":
            msg = "यह छवि एक श्वेत-श्याम रेखाचित्र या स्केच प्रतीत होती है, न कि असली पौधे की पत्ती।"
            recoms = [
                "सुनिश्चित करें कि फोटो एक वास्तविक तस्वीर है (चित्र या स्केच नहीं)।",
                "सुनिश्चित करें कि पत्ती स्पष्ट रूप से दिखाई दे रही है।",
                "सुनिश्चित करें कि पौधा आलू की फसल है।",
            ]
        elif lang == "mr":
            msg = "ही प्रतिमा एक कृष्णधवल रेखाचित्र किंवा स्केच वाटते, खरी वनस्पतीची पाने नाही."
            recoms = [
                "फोटो प्रत्यक्ष छायाचित्र असल्याचे सुनिश्चित करा (रेखाचित्र किंवा कलाकृती नाही).",
                "पाने स्पष्टपणे दिसत असल्याची खात्री करा.",
                "वनस्पती बटाट्याचे पीक असल्याची खात्री करा.",
            ]
        else:
            msg = "The image appears to be a black & white drawing or sketch, not a real plant leaf."
            recoms = [
                "Ensure the photo is a real photograph (not a drawing or artwork).",
                "Ensure the leaf is clearly visible.",
                "Ensure the plant is a potato crop.",
            ]
        return {"message": msg, "recommendations": recoms}

    if disease not in translations:
        return {"message": "Unknown disease", "recommendations": []}

    return {
        "message": translations[disease].get(lang, translations[disease]["en"]),
        "recommendations": recommendations[disease].get(
            lang, recommendations[disease]["en"]
        ),
    }

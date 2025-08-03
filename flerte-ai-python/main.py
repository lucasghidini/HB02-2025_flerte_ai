from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from models import ConversationRequest
from gemini_service import GeminiService
from config import settings
import uvicorn
import json
from PIL import Image
import io

app = FastAPI(
    title="Flert-AI API",
    description="API com personalização de preferências do usuário",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gemini_service = GeminiService()

def optimize_image(image_bytes: bytes) -> bytes:
    try:
        max_size = (1024, 1024)
        buffer = io.BytesIO(image_bytes)
        img = Image.open(buffer)
        
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG", quality=85)
        return output_buffer.getvalue()
    except Exception as e:
        print(f"Erro ao otimizar a imagem: {e}")
        return image_bytes

@app.post("/chat")
async def chat(request: ConversationRequest):
    try:
        if not any(msg.role.lower() == "user" and msg.content for msg in request.history):
            raise HTTPException(status_code=400, detail="Nenhuma mensagem de usuário encontrada no histórico")

        conversation_history = []
        for msg in request.history:
            conversation_history.append({
                "role": msg.role.lower(),
                "content": msg.content
            })
        
        response_text = await gemini_service.generate_response_with_context(
            history=conversation_history,
            preferences=request.preferences
        )
        
        return {
            "response": response_text,
            "applied_preferences": {
                "style": request.preferences.style,
                "length": request.preferences.length
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.post("/chat/with-image")
async def chat_with_image(
    image: UploadFile = File(...),
    request_data: str = Form(...)
):
    try:
        request = ConversationRequest.model_validate_json(request_data)

        image_bytes = await image.read()
        optimized_image_bytes = optimize_image(image_bytes)
        
        user_message_found = False
        for msg in reversed(request.history):
            if msg.role.lower() == "user":
                msg.image_data = optimized_image_bytes
                user_message_found = True
                break
        
        if not user_message_found:
            raise HTTPException(status_code=400, detail="Nenhuma mensagem de usuário encontrada no histórico para anexar a imagem.")

        conversation_history = []
        for msg in request.history:
            history_item = {
                "role": msg.role.lower(),
                "content": msg.content
            }
            if msg.image_data:
                history_item["image_data"] = msg.image_data
            conversation_history.append(history_item)
            
        response_text = await gemini_service.generate_response_with_context(
            history=conversation_history,
            preferences=request.preferences
        )
        
        return {
            "response": response_text,
            "applied_preferences": {
                "style": request.preferences.style,
                "length": request.preferences.length
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


if __name__ == "__main__":
    uvicorn
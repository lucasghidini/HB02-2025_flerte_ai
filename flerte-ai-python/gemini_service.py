import google.generativeai as genai
from config import settings
from models import UserPreferences
from PIL import Image
import io

class GeminiService:
    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY não encontrada")
        
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    
    async def generate_response_with_context(
        self,
        history: list,
        preferences: UserPreferences
    ) -> str:
        try:
            content_parts = []
            
            system_prompt = f"""
            Você é Flert-AI, um assistente especialista em relacionamentos e comunicação.
            
            CONFIGURAÇÕES DE PERSONALIZAÇÃO:
            - Estilo: {preferences.style}
            - Tamanho: {preferences.length}
            
            DIRETRIZES:
            - Forneça conselhos práticos e respeitosos
            - Promova comunicação saudável e consensual
            - Adapte o tom conforme o estilo definido ({preferences.style})
            - Respeite o tamanho de resposta solicitado ({preferences.length})
            """
            
            content_parts.append(system_prompt)

            for msg in history:
                content_parts.append(msg['content'])

                if 'image_data' in msg and msg['image_data']:
                    try:
                        image = Image.open(io.BytesIO(msg['image_data']))
                        content_parts.append(image)
                    except Exception as img_e:
                        print(f"Erro ao processar imagem: {str(img_e)}")
            
            response = self.model.generate_content(content_parts)
            return response.text
            
        except Exception as e:
            raise Exception(f"Erro ao gerar resposta do Gemini: {str(e)}")
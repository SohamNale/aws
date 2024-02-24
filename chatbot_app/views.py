# chatbot_project/chatbot_app/views.py
from django.shortcuts import render
from django.http import JsonResponse
from .models import Chatbot, ChatbotTraining

# Initialize training instance
# Usage Example:
translation_model_name = "Helsinki-NLP/opus-mt-en-es"
source_language_code = "es"
target_language_code = "en"
model_name = "anakin87/zephyr-7b-alpha-sharded"
folder_path = '/content/drive/MyDrive/chatbot_data/'
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

chatbot_training = ChatbotTraining(translation_model_name, source_language_code, target_language_code, model_name, folder_path, embedding_model_name)

# Store chatbot instance globally for simplicity
chatbot_instance = Chatbot(translation_model_name, source_language_code, target_language_code,
                          chatbot_training.model, chatbot_training.tokenizer,
                          chatbot_training.retriever, chatbot_training.llm, [])

def home(request):
    return render(request, 'home.html')

def chat(request):
    if request.method == 'POST':
        data = request.json()
        query = data.get('query', '')
        use_spanish = data.get('use_spanish', '') == 'true'

        if use_spanish:
            translated_query = chatbot_instance.translate(query)
        else:
            translated_query = query

        response, chat_history = chatbot_instance.create_conversation(translated_query)

        if use_spanish:
            translated_response = chatbot_instance.translate(response)
        else:
            translated_response = response

        return JsonResponse({'response': translated_response, 'chat_history': chat_history})

    return render(request, 'chat.html')

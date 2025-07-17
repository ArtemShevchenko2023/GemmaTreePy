import logging
import telebot
import json
from datetime import datetime
from telebot.types import Message
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import token
# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
MODEL_NAME = "google/gemma-3-1b-it"
MAX_HISTORY = 3
LOG_FILE = "user_requests.json"

# Настройки генерации
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_k": 20,
    "top_p": 0.9,
    "do_sample": True,
    "max_new_tokens": 4096,
    "repetition_penalty": 1.1,
    "num_return_sequences": 1
}

# Инициализация модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    logger.info("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    logger.info("Загрузка модели...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto" if device.type == 'cuda' else None,
        low_cpu_mem_usage=True
    )
    
    if device.type == 'cpu':
        model = model.to(device)
        model = model.eval()
        torch.set_num_threads(2)
    
    logger.info(f"Модель загружена на {device}!")
except Exception as e:
    logger.error(f"Ошибка при загрузке модели: {e}")
    exit()

# Хранилище контекста
chat_histories = {}

def save_interaction(user_id: int, username: str, user_message: str, ai_response: str):
    """Сохраняет взаимодействие пользователя и ИИ в JSON файл"""
    try:
        record = {
            "user_id": user_id,
            "username": username,
            "user_message": user_message,
            "ai_response": ai_response,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        
        data.append(record)
        
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"Ошибка при сохранении запроса: {e}")

def format_history(history: list) -> list:
    """Форматирует историю для модели"""
    return [{"role": "user" if msg["is_user"] else "assistant", "content": msg["content"]} 
            for msg in history[-MAX_HISTORY:]]

def generate_response(prompt: str, history: list) -> str:
    """Генерирует ответ с учетом истории"""
    try:
        history.append({"is_user": True, "content": prompt})
        formatted_history = format_history(history)
        
        inputs = tokenizer.apply_chat_template(
            formatted_history,
            add_generation_prompt=True,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)
        
        outputs = model.generate(
            inputs,
            **GENERATION_CONFIG
        )
        
        response = tokenizer.decode(
            outputs[0][inputs.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        history.append({"is_user": False, "content": response})
        return response
        
    except Exception as e:
        logger.error(f"Ошибка генерации: {e}")
        return "⚠️ Извините, не удалось обработать запрос. Попробуйте короче или позже."

# Инициализация бота
bot = telebot.TeleBot(token)

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message: Message):
    """Обработка команды /start - приветственное сообщение"""
    welcome_text = (
        "Привет! 👋\n\n"
        "Я - умный бот с искусственным интеллектом.\n"
        "Просто напиши мне сообщение, и я постараюсь помочь!\n\n"
        "Доступные команды:\n"
        "/newchat - начать новый диалог (очистить контекст)\n"
        "/clearchat - очистить историю текущего чата\n\n"
        "Все диалоги сохраняются для улучшения качества сервиса."
    )
    bot.reply_to(message, welcome_text)

# Обработчик команды /newchat
@bot.message_handler(commands=['newchat'])
def new_chat(message: Message):
    """Начинает новый диалог (очищает историю)"""
    chat_id = message.chat.id
    if chat_id in chat_histories:
        del chat_histories[chat_id]
    bot.reply_to(message, "🆕 Новый диалог начат! Контекст очищен.")

# Обработчик команды /clearchat
@bot.message_handler(commands=['clearchat'])
def clear_chat(message: Message):
    """Очищает историю текущего чата"""
    chat_id = message.chat.id
    if chat_id in chat_histories:
        chat_histories[chat_id] = []
        bot.reply_to(message, "🧹 История текущего чата очищена!")
    else:
        bot.reply_to(message, "История чата уже пуста!")

# Обработчик всех остальных текстовых сообщений
@bot.message_handler(func=lambda message: True)
def handle_message(message: Message):
    """Обработка всех входящих сообщений (кроме команд)"""
    try:
        # Пропускаем команды (начинаются с /)
        if message.text.startswith('/'):
            return
            
        chat_id = message.chat.id
        user_id = message.from_user.id
        username = message.from_user.username or f"user_{user_id}"
        
        if chat_id not in chat_histories:
            chat_histories[chat_id] = []
        
        # Показываем статус "печатает"
        bot.send_chat_action(chat_id, 'typing')
        
        # Генерируем ответ
        response = generate_response(message.text, chat_histories[chat_id])
        
        # Сохраняем взаимодействие
        save_interaction(user_id, username, message.text, response)
        
        # Отправляем ответ
        bot.reply_to(message, response)
        
    except Exception as e:
        logger.error(f"Ошибка обработки сообщения: {e}")
        bot.reply_to(message, "⚠️ Произошла внутренняя ошибка. Попробуйте позже.")

if __name__ == "__main__":
    logger.info("Бот запущен! Логи сохраняются в %s", LOG_FILE)
    bot.infinity_polling()
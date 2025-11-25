import os
import uuid
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from core.config import settings
from middleware.auth.database import SessionLocal
from middleware.auth.models import Users
from services import storage_service
from helper.ocr import run_prediction

# Configure logging for the bot
logger = logging.getLogger(__name__)

# Initialize the Application builder (we will build and start it in main.py)
ptb_application = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()

def get_db_session():
    return SessionLocal()

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for the /start command.
    Checks if the Telegram user is linked to a database user.
    """
    chat_id = update.effective_chat.id
    db = get_db_session()
    
    try:
        user = db.query(Users).filter(Users.telegram_chat_id == chat_id).first()
        
        if user:
            await update.message.reply_text(f"Welcome back, {user.username}! Send me an image to extract text.")
        else:
            # Generate the linking URL
            # The token is simply the chat_id. In production, sign this token!
            link = f"{settings.FRONTEND_URL}/connect-telegram?token={chat_id}"
            
            await update.message.reply_text(
                f"👋 You are not connected to an account.\n\n"
                f"1. Log in to the OCR website.\n"
                f"2. Click this link to connect your account:\n{link}\n\n"
                f"Once connected, you can send images here!"
            )
    finally:
        db.close()

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for photo messages.
    Downloads photo -> Uploads to S3 -> Runs OCR -> Replies with text.
    """
    chat_id = update.effective_chat.id
    db = get_db_session()
    
    try:
        # 1. Auth Check
        user = db.query(Users).filter(Users.telegram_chat_id == chat_id).first()
        if not user:
            await update.message.reply_text("⚠️ Please /start and connect your account first.")
            return

        # 2. Inform user processing started
        status_msg = await update.message.reply_text("⏳ Processing image...")

        # 3. Download File from Telegram
        photo_file = await update.message.photo[-1].get_file()
        
        job_id = str(uuid.uuid4())
        filename = f"{job_id}.jpg"
        local_dir = "./temp_downloads"
        local_path = os.path.join(local_dir, filename)
        os.makedirs(local_dir, exist_ok=True)
        
        await photo_file.download_to_drive(local_path)

        try:
            # 4. Upload to S3 (Standardize storage path)
            s3_key = f"uploads/telegram/{chat_id}/{filename}"
            
            # Using boto3 client directly from storage_service to upload file object
            with open(local_path, "rb") as f:
                 storage_service.s3_client.upload_fileobj(
                     f, 
                     settings.SPACES_NAME, 
                     s3_key,
                     ExtraArgs={'ContentType': 'image/jpeg', 'ACL': 'private'}
                 )

            # 5. Run OCR Inference
            # Ensure your run_prediction function accepts these arguments
            text = run_prediction(
                settings.YOLO_MODEL_PATH, 
                settings.CRNN_MODEL_PATH, 
                local_path,
                settings.OCR_MODE
            )

            # 6. Reply with result
            if text.strip():
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_msg.message_id,
                    text=f"✅ **Result:**\n\n{text}",
                    parse_mode="Markdown"
                )
            else:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_msg.message_id,
                    text="❌ No text detected."
                )

        except Exception as e:
            logger.error(f"OCR Error: {e}")
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_msg.message_id,
                text="❌ An error occurred during processing."
            )
        
        finally:
            # Cleanup local file
            if os.path.exists(local_path):
                os.remove(local_path)

    finally:
        db.close()

# Register handlers to the application
ptb_application.add_handler(CommandHandler("start", start_command))
ptb_application.add_handler(MessageHandler(filters.PHOTO, handle_image))
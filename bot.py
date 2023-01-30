import logging
import os

from telegram import __version__ as TG_VER
from GPT2.conv_ai_model_ja import ConvAIModelJa

from persona_captiopn import PersonaCaption

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 5):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

PHOTO = 0
TOKEN = os.environ.get("TOKEN")
PERSONA_LIST = []
DIALOG_HISTORY = []
CHAT_END = False

# We used 5 pairs of personas in the dialogue model training, so output num is 5.
PERSONA_OUTPUT_NUM = 5

CONV_AI_PARAMS = {
    "do_sample": True,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.9,
    "max_history": 3,
    "max_length": 50,
    "min_length": 10,
}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the conversation and asks the user about their gender."""
    global PERSONA_LIST, DIALOG_HISTORY, CHAT_END
    PERSONA_LIST = []
    DIALOG_HISTORY = []
    CHAT_END = False

    await update.message.reply_text(
        "こんにちは、ペルソナ対話ボットです。\n私は送信された人物画像になりきってチャットを行います。\n\n"
        "ボットのペルソナとして設定したい人物の画像を送信してください。\n"
        "/skip コマンドで画像の送信をスキップし、デフォルトのボットとしてチャットすることも可能です。",
        reply_markup=ReplyKeyboardRemove(),
    )
    return PHOTO


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the photo."""
    user = update.message.from_user
    photo_file = await update.message.photo[-1].get_file()
    image_path = "./photo/portrait.jpg"
    await photo_file.download_to_drive(image_path)
    logger.info("Photo of %s: %s", user.first_name, image_path)

    persona_caption = PersonaCaption()
    global PERSONA_LIST
    PERSONA_LIST = persona_caption.get_caption(image_path, PERSONA_OUTPUT_NUM)
    assert PERSONA_OUTPUT_NUM == len(PERSONA_LIST)

    await update.message.reply_text(
        "ありがとうございます。この人物のペルソナは以下になります。\n\n"
        + "----------\n"
        + "\n".join(PERSONA_LIST)
        + "\n----------\n\n"
        "以上をボットのペルソナとして設定します。\n\n"
        "チャットを始めましょう!"
    )

    return ConversationHandler.END


async def skip_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Skips the photo."""
    user = update.message.from_user
    logger.info("User %s did not send a photo.", user.first_name)
    await update.message.reply_text(
        "画像の送信をスキップしました。\n" "デフォルトのボットとして設定します。\n\nチャットを始めましょう!"
    )

    return ConversationHandler.END


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Chat using gpt2 model."""
    user = update.message.from_user

    global CHAT_END
    if not CHAT_END:
        user_message = update.message.text

        model = ConvAIModelJa("./GPT2/model/", args=CONV_AI_PARAMS)
        global DIALOG_HISTORY, PERSONA_LIST
        reply, DIALOG_HISTORY = model.interact_single(
            user_message, history=DIALOG_HISTORY, personality=PERSONA_LIST
        )

        logger.info("User %s send 「%s」 to bot", user.first_name, user_message)
        logger.info("Length of dialog history is %s.", len(DIALOG_HISTORY))
        await update.message.reply_text(reply)
    else:
        await update.message.reply_text("チャットを始めたい場合は/startコマンドを入力してください。")


async def goodbye(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    logger.info("User %s ended the conversation.", user.first_name)
    await update.message.reply_text(
        "会話を終了します。\nまたチャットを始めたい場合は/startコマンドを入力してください。\n\n" "さようなら！",
        reply_markup=ReplyKeyboardRemove(),
    )
    # re-initialize
    global PERSONA_LIST, DIALOG_HISTORY, CHAT_END
    PERSONA_LIST = []
    DIALOG_HISTORY = []
    CHAT_END = True
    return ConversationHandler.END


def main() -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # Add conversation handler with the states GENDER, PHOTO, LOCATION and BIO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            PHOTO: [
                MessageHandler(filters.PHOTO, photo),
                CommandHandler("skip", skip_photo),
            ],
        },
        fallbacks=[CommandHandler("goodbye", goodbye)],
    )

    application.add_handler(conv_handler)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    application.add_handler(CommandHandler("goodbye", goodbye))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()

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

PHOTO, CHAT = range(2)
TOKEN = os.environ.get("TOKEN")
PERSONA_LIST = []
DIALOG_HISTORY = []

# Dialogue model training was used JPersonaChat(5 pairs of personas), so output num is 5.
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
    user = update.message.from_user
    logger.info("User %s activated this bot", user.first_name)

    global PERSONA_LIST, DIALOG_HISTORY
    PERSONA_LIST = []
    DIALOG_HISTORY = []

    await update.message.reply_text(
        "こんにちは、ペルソナ対話ボットです。\n私は送信された人物のペルソナに沿ってチャットを行います。\n\n"
        "ボットのペルソナとして設定したい人物の画像を送信してください。\n"
        "/skip コマンドで画像の送信をスキップし、ランダムに選択されたペルソナを持つボットとしてチャットすることも可能です。",
        reply_markup=ReplyKeyboardRemove(),
    )
    return PHOTO


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the photo."""
    user = update.message.from_user
    photo_file = await update.message.photo[-1].get_file()

    if not os.path.exists("./photo"):
        os.mkdir("./photo")

    image_path = "./photo/portrait.jpg"
    await photo_file.download_to_drive(image_path)
    logger.info("Photo of User %s: %s", user.first_name, image_path)

    persona_caption = PersonaCaption()
    global PERSONA_LIST
    PERSONA_LIST = persona_caption.get_persona_list(image_path, PERSONA_OUTPUT_NUM)
    assert PERSONA_OUTPUT_NUM == len(PERSONA_LIST)

    await update.message.reply_text(
        "ありがとうございます。この人物のペルソナは以下になります。\n\n"
        + "----------\n"
        + "\n".join(PERSONA_LIST)
        + "\n----------\n\n"
        "以上をボットのペルソナとして設定します。\n\n"
        "チャットを始めましょう!"
    )

    return CHAT


async def skip_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Skips the photo."""
    user = update.message.from_user
    logger.info("User %s did not send a photo.", user.first_name)

    persona_caption = PersonaCaption()
    global PERSONA_LIST
    PERSONA_LIST = persona_caption.get_random_persona_list(PERSONA_OUTPUT_NUM)
    assert PERSONA_OUTPUT_NUM == len(PERSONA_LIST)

    await update.message.reply_text(
        "画像の送信をスキップしました。\n"
        "ランダムに選択されたペルソナは以下になります。\n\n"
        + "----------\n"
        + "\n".join(PERSONA_LIST)
        + "\n----------\n\n"
        "以上をボットのペルソナとして設定します。\n\n"
        "チャットを始めましょう!"
    )

    return CHAT


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Chat using gpt2 model."""
    user = update.message.from_user

    user_message = update.message.text

    if not os.path.exists("./GPT2/model") or not os.path.isfile(
        "./GPT2/model/pytorch_model.bin"
    ):
        logger.error(
            "GPT2 model file pytorch_model.bin is not placed under ./GPT2/model"
        )
        await update.message.reply_text("内部エラーが発生しました。\n現在チャットを行うことができません。")
        return ConversationHandler.END

    model = ConvAIModelJa("./GPT2/model/", args=CONV_AI_PARAMS)
    global DIALOG_HISTORY, PERSONA_LIST
    reply, DIALOG_HISTORY = model.interact_single(
        user_message, history=DIALOG_HISTORY, personality=PERSONA_LIST
    )

    logger.info("User %s send 「%s」 to bot", user.first_name, user_message)
    await update.message.reply_text(reply)


async def goodbye(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    logger.info("User %s ended the conversation.", user.first_name)
    await update.message.reply_text(
        "会話を終了します。\nまたチャットを始めたい場合は/startコマンドを入力してください。\n\n" "さようなら！",
        reply_markup=ReplyKeyboardRemove(),
    )
    # re-initialize
    global PERSONA_LIST, DIALOG_HISTORY
    PERSONA_LIST = []
    DIALOG_HISTORY = []
    return ConversationHandler.END


def main() -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            PHOTO: [
                MessageHandler(filters.PHOTO, photo),
                CommandHandler("skip", skip_photo),
            ],
            CHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, chat)],
        },
        fallbacks=[CommandHandler("goodbye", goodbye)],
    )

    application.add_handler(conv_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()

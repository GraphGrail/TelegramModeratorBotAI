import os
import telebot
import sqlite3
import settings
from time import time
from datetime import datetime
from commentEvaluator import CommentEvaluator
from keras.models import load_model
import pickle

#spacy.load('xx_ent_wiki_sm')
script_dir = os.path.dirname(__file__)
bot = telebot.TeleBot(settings.BOT_TOKEN)
conn = sqlite3.connect(os.path.join(script_dir, 'universabot.db'), check_same_thread=False)
c = conn.cursor()

with open(os.path.join(script_dir, 'model_files/token.pickle'), 'rb') as handle:
    tok = pickle.load(handle)

model = load_model(os.path.join(script_dir, 'model_files/rus_weights2.hdf5'))
evaluator = CommentEvaluator(model, tok)


def get_language(lang_code):
    if not lang_code:
        return "en"
    if "-" in lang_code:
        lang_code = lang_code.split("-")[0]
    if lang_code == "ru":
        return "ru"
    else:
        return "en"


@bot.message_handler(func=lambda message: message.entities is not None)
def delete_links(message):
    for entity in message.entities:
        if entity.type in settings.RESTRICTED_LINKS:
            bot.delete_message(message.chat.id, message.message_id)
            bot.send_message(message.chat.id, settings.strings.get(get_language(message.from_user.language_code)).get("ro_link"), reply_to_message_id=message.message_id)
            bot.restrict_chat_member(message.chat.id, message.from_user.id, until_date=time()+60*settings.MINUTES)
        else:
            return


@bot.message_handler(func=lambda message: message.text and evaluator.analyze(message.text) == "negative")
def handle_negative(message):

    c.execute('SELECT ID FROM Userstat WHERE user_id=? and chat_id=?', (int(message.from_user.id),
                                                                        int(message.chat.id)))
    rows = c.fetchall()
    if len(rows) == 0:
        c.execute('INSERT INTO Userstat VALUES(NULL, ?, ?, ?, ?, ?, ?)', (message.from_user.id,
                                                                          message.from_user.first_name, 1,
                                                                          datetime.now(), 0, message.chat.id,))
        c.execute('INSERT INTO message VALUES(NULL, ?, ?, ?, ?, ?, ?)', (message.message_id, message.text, 0,
                                                                         message.from_user.id, datetime.now(),
                                                                         message.chat.id,))
        conn.commit()
    else:
        c.execute('UPDATE Userstat SET negative_total=negative_total+1')
        c.execute('Insert INTO message VALUES(NULL, ?, ?, ?, ?, ?, ?)', (message.message_id, message.text, 0,
                                                                         message.from_user.id, datetime.now(),
                                                                         message.chat.id,))
        conn.commit()
        c.execute("SELECT MESSAGE_ID FROM message WHERE CREATED_AT >= Datetime('now', '-60 minutes', 'localtime') AND SENITMENT=0")
        rows = c.fetchall()
        print(rows)
        if len(rows) >= 5:
            bot.restrict_chat_member(message.chat.id, message.from_user.id, until_date=time()+60*settings.MINUTES)
            bot.send_message(message.chat.id,
                             settings.strings.get(get_language(message.from_user.language_code)).get("ro_msg"),
                             reply_to_message_id=message.message_id)
            c.execute("INSERT INTO banned VALUES(NULL, ?, Datetime('now'), Datetime('now', '+180 minutes', 'localtime'))", message.from_user.id)
            conn.commit()
            #bot.delete_message(message.chat.id, message.message_id)


@bot.message_handler(func=lambda message: message.text and evaluator.analyze(message.text) == "positive")
def handle_positive(message):
    c.execute('SELECT ID FROM Userstat WHERE user_id=? and chat_id=?',
              (int(message.from_user.id), int(message.chat.id)))
    rows = c.fetchall()
    if len(rows) == 0:
        c.execute('INSERT INTO Userstat VALUES(NULL, ?, ?, ?, ?, ?, ?)', (message.from_user.id,
                                                                          message.from_user.first_name, 0,
                                                                          datetime.now(), 1, message.chat.id,))
        c.execute('INSERT INTO message VALUES(NULL, ?, ?, ?, ?, ?, ?)', (message.message_id, message.text, 1,
                                                                         message.from_user.id, datetime.now(),
                                                                         message.chat.id,))
        conn.commit()
    else:
        c.execute('UPDATE Userstat SET positive_total=positive_total+1')
        c.execute('Insert INTO message VALUES(NULL, ?, ?, ?, ?, ?, ?)', (message.message_id, message.text, 1,
                                                                         message.from_user.id, datetime.now(),
                                                                         message.chat.id,))
        conn.commit()

@bot.message_handler(func=lambda message: message.text and evaluator.analyze(message.text) == "neutral")
def handle_positive(message):
    c.execute('SELECT ID FROM Userstat WHERE user_id=? and chat_id=?',
              (int(message.from_user.id), int(message.chat.id)))
    rows = c.fetchall()
    if len(rows) == 0:
        c.execute('INSERT INTO Userstat VALUES(NULL, ?, ?, ?, ?, ?, ?)', (message.from_user.id,
                                                                          message.from_user.first_name, 0,
                                                                          datetime.now(), 0, message.chat.id,))
        c.execute('INSERT INTO message VALUES(NULL, ?, ?, ?, ?, ?, ?)', (message.message_id, message.text, 2,
                                                                         message.from_user.id, datetime.now(),
                                                                         message.chat.id,))
        conn.commit()
    else:
        c.execute('UPDATE Userstat SET positive_total=positive_total+1')
        c.execute('Insert INTO message VALUES(NULL, ?, ?, ?, ?, ?, ?)', (message.message_id, message.text, 2,
                                                                         message.from_user.id, datetime.now(),
                                                                         message.chat.id,))
        conn.commit()


if __name__ == "__main__":
    bot.polling(none_stop=True)

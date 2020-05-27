from predict import pred
import telebot
from user_statistic import Stat

bot = telebot.TeleBot('1003593379:AAEXv1sd57DbWDh0m7u-eQQSz3MYaddCVzE')
pred_result = None
pred_stats = None

user_data = Stat('SeasUp', 0, 0, 0, 0, 0, 0)


@bot.message_handler(commands=['start'])
def send_welcome(message):
	# bot.reply_to(message, "Добро пожаловать! Пока что я не умею ничего, но это ненадолго\n
	# Используйте для помощи:\n/help")
	bot.send_message(message.from_user.id,
					 'Добро пожаловать! Этот бот создан для диагностики развития сердечнососудистых заболеваний '
					 'в ближайшие 10 лет\nИспользуйте для помощи:\n/help')


@bot.message_handler(commands=['help'])
def send_help(message):
	bot.send_message(message.from_user.id,
					 'Список команд:\n/help: помощь\n/predict: начать диагностику')


@bot.message_handler(commands=['predict'])
def prediction(message):
	# a = message.text
	# a.split()
	# print(a)
	bot.send_message(message.from_user.id, 'Для того, чтобы начать диагностику, следует сначала получить анализы '
										   'крови на количесво холестерина и сахара в крови')
	bot.send_message(message.from_user.id, 'Итак, приступим!\nДля начала введите ваш возраст:')
	bot.register_next_step_handler(message, get_user_age)


def get_user_age(message):
	'''global pred_stats
	pred_stats = message.text
	pred_stats = pred_stats.split()
	pred_stats = [float(i) for i in pred_stats]
	global pred_result
	pred_result = pred(pred_stats)'''
	global user_data
	try:
		user_data.get_age(int(message.text))
	except:
		pass
	bot.send_message(message.from_user.id, 'Теперь введите ваш пол:')
	bot.register_next_step_handler(message, get_user_sex)


def get_user_sex(message):
	global user_data
	try:
		user_data.get_sex(int(message.text))
	except:
		pass
	# bot.register_next_step_handler(message, get_user_sex())
	# bot.send_message(message.from_user.id, user_data.send_back()[1])
	print(*user_data.send_back())
	bot.send_message(message.from_user.id, 'Отлично! Если вы курите, то введите количество сигарет, которые вы '
										   'выкуриваете за день(0, если не курите)')
	bot.register_next_step_handler(message, get_user_cigs)


def get_user_cigs(message):
	global user_data
	try:
		user_data.get_cigs(int(message.text))
	except:
		pass
	bot.send_message(message.from_user.id, 'Введите общий уровень холестерина в крови')
	bot.register_next_step_handler(message, get_user_chol)


def get_user_chol(message):
	global user_data
	try:
		user_data.get_chol(float(message.text))
	except:
		pass
	bot.send_message(message.from_user.id, 'Принимаете ли вы в данный момент препараты, которые влияют на ваше '
										   'кровяное давление?')
	bot.register_next_step_handler(message, get_user_bp)


def get_user_bp(message):
	global user_data
	try:
		user_data.get_bp(int(message.text))
	except:
		pass
	bot.send_message(message.from_user.id, 'Введите общий уровень глюкозы в крови')
	bot.register_next_step_handler(message, get_user_glucose)


def get_user_glucose(message):
	global user_data
	try:
		user_data.get_glucose(float(message.text))
	except:
		pass
	bot.send_message(message.from_user.id, 'Отлично, все почти готово!')
	# bot.register_next_step_handler(message, get_user_results)
	#bot.send_message(message.from_user.id, *user_data.predict_result())
	print(user_data.predict_result()[0])
	if user_data.predict_result()[0] == 0:
		bot.send_message(message.from_user.id, 'С большой вероятностью риск развития ишемической болезни сердца '
											   'у вас отсутствует')
	if user_data.predict_result()[0] == 1:
		bot.send_message(message.from_user.id, 'Предварительный анализ показывает что у вас есть риск развития '
											   'ишемической болезни сердца')
	bot.send_message(message.from_user.id, 'Для уточнения результатов обязательно проконсультируйтесь со специалистом')

'''def get_user_results(message):
	bot.send_message(message.from_user.id, *user_data.predict_result())
	print(*user_data.predict_result())'''


@bot.message_handler(func=lambda m: True)
def echo_all(message):
	bot.reply_to(message, 'excuse me, what is ' + message.text + '?')


bot.polling()

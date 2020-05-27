from predict import pred


class Stat:
	def __init__(self, version, age, sex, cigs, chol, bp, glucose):
		self.version = version
		self.age = age
		self.sex = sex
		self.cigs = cigs
		self.chol = chol
		self.bp = bp
		self.glucose = glucose

	def get_age(self, age):
		self.age = age

	def get_sex(self, sex):
		self.sex = sex

	def get_cigs(self, cigsPerDay):
		self.cigs = cigsPerDay

	def get_chol(self, totChol):
		self.chol = int(totChol * 38.665)

	def get_bp(self, sysBP):
		self.bp = sysBP

	def get_glucose(self, glucose):
		self.glucose = int(glucose * 18.016)

	def predict_result(self):
		return pred(([self.age, self.sex, self.cigs, self.chol, self.bp, self.glucose]), 'true_seasup.pkl')

	def send_back(self):
		return [self.age, self.sex]

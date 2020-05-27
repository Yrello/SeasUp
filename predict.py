def pred(inp, modelname):
	import joblib
	import numpy as np
	try:
		from sklearn.externals import joblib
	except:
		pass

	new_model = joblib.load(modelname)
	a = np.array(inp)
	a = a.reshape(1, -1)
	# print('Tree prediction:',*model.predict(a))
	# print('XGBoot prediction:',*new_model.predict(a))
	return new_model.predict(a)


class Hparams:
	def __init__(self):
		### data and save path
		self.train_record_path = 'data/data.train'
		self.threshold = 0.5
		
		self.num_train_sample = 8300 # 37000
		#self.valid_record_path = 'preprocess/data/record/data_4000_400.test'
		self.valid_record_path = 'data/data.val'

		self.num_valid_sample = 200
		
		self.save_path = 'model_512Unit' 

		self.save_best = False
		self.max_to_keep = 1000

		### input params
		self.max_width = 800
		self.max_height = 800
		self.num_point = 4

		
		
		self.dense_units = 512
		self.weight_decay = 0.00004

		### attention params
		### training params
		self.batch_size = 1
		self.max_epochs = 600
		self.lr = 0.0001 #Tuning hyperparameters

hparams = Hparams()

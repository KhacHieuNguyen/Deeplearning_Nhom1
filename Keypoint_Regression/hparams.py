
class Hparams:
	def __init__(self):
		### data and save path
		self.train_record_path = 'data/data_gen.train'

		
		self.num_train_sample = 8300 # 37000
		#self.valid_record_path = 'preprocess/data/record/data_4000_400.test'
		self.valid_record_path = 'data/data_gen.val'

		self.num_valid_sample = 200# 450
		
		self.save_path = 'model_512Unit' 
		#self.save_path = 'models/mobile_net'
		#self.save_path = 'b1_0709_beta_0'
		#self.save_path = 'B1_efficient_300_400_layer7_b16_lambda0_14point'
		#self.save_path = 'efficient_4000_400_layer5_b16_lambda1_mse_aug'
		#self.save_path = 'training_checkpoints_efficient_400_linear_nodense_b16_lambda0_mse-linear_400-linear_400'
		#self.save_path = 'training_checkpoints_mobile_400_linear_nodense_b32_lambda5_mse'# 'training_checkpoints_mobile_400_no_dense'
		#self.save_path = 'training_checkpoints_mobile_400_linear_dense_b32_lambda1_mse'
		self.save_best = False
		self.max_to_keep = 1000

		### input params
		self.max_width = 400# (600, 600, 3)
		self.max_height = 644
		self.num_point = 4

		### conv_tower params
		# base model from tf.keras.application, or custom instance of tf.keras.Model
		# check for new models from https://www.tensorflow.org/api_docs/python/tf/keras/applications
		# check for newest model from tf-nightly version
		#self.base_model_name = 'EfficientNetB0'
		self.base_model_name = 'EfficientNetB0'
		# self.base_model_name = 'EfficientNetB1'
		#self.base_model_name = 'mobinet' # 'mobinet'
		# last convolution layer from base model which extract features from
		# inception v3: mixed2 (mixed_5d in tf.slim inceptionv3)
		# inception resnet v2: (mixed_6a in tf.slim inception_resnet_v2)
		self.end_point ='block7a_se_squeeze'#'block5c_se_squeeze'# 'block7a_se_squeeze'#'conv_pw_13_relu' #Tuning Hyperparameters
		#self.end_point = 'conv_pw_13_relu' #for mobilenet
		# self.end_point = 'block7b_se_squeeze' # 'block6a_se_squeeze'#'block7b_se_squeeze'
		
		self.dense_units = 512
		self.weight_decay = 0.00004

		### attention params
		### training params
		self.batch_size = 16
		self.max_epochs = 600
		self.lr = 0.0001 #Tuning hyperparameters

hparams = Hparams()

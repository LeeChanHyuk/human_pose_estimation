from msilib.schema import Class
from data.stabilizer import Stabilizer
import numpy as np

class StabilizersWithKalmanFilter():
	def __init__(self) -> None:
		self.center_eye_stabilizers = [Stabilizer(
			state_num=2,
			measure_num=1,
			cov_process=0.1,
			cov_measure=0.1
		) for i in range(3)]
		self.left_shoulder_stabilizers = [Stabilizer(
			state_num=2,
			measure_num=1,
			cov_process=0.1,
			cov_measure=0.1
		) for i in range(3)]
		self.right_shoulder_stabilizers = [Stabilizer(
			state_num=2,
			measure_num=1,
			cov_process=0.1,
			cov_measure=0.1
		) for i in range(3)]
		self.center_mouth_stabilizers = [Stabilizer(
			state_num=2,
			measure_num=1,
			cov_process=0.1,
			cov_measure=0.1
		) for i in range(3)]
		self.center_stomach_stabilizers = [Stabilizer(
			state_num=2,
			measure_num=1,
			cov_process=0.1,
			cov_measure=0.1
		) for i in range(3)]
		self.head_pose_stabilizers = [Stabilizer(
			state_num=2,
			measure_num=1,
			cov_process=0.1,
			cov_measure=0.1
		) for i in range(3)]
		self.body_pose_stabilizers = [Stabilizer(
			state_num=2,
			measure_num=1,
			cov_process=0.1,
			cov_measure=0.1
		) for i in range(3)]
		self.eye_pose_stabilizers = [Stabilizer(
			state_num=2,
			measure_num=1,
			cov_process=0.1,
			cov_measure=0.1
		) for i in range(4)]

dummy_list = [[1.0, 1.0, 1.0] for x in range(200)]

class HumanInfo():
	def __init__(self) -> None:
		# Positions of human
		self.center_eyes = [[1.0, 1.0, 1.0] for x in range(200)]
		self.center_mouths = [[-100.0, -100.0, -100.0] for x in range(200)]
		self.left_shoulders = [[-100.0, -100.0, -100.0] for x in range(200)]
		self.right_shoulders = [[-100.0, -100.0, -100.0] for x in range(200)]
		self.center_stomachs = [[-100.0, -100.0, -100.0] for x in range(200)]
		self.face_box = [1.0, 1.0, 1.0, 1.0]
		self.left_eye_box = [1.0, 1.0, 1.0, 1.0]
		self.right_eye_box = [1.0, 1.0, 1.0, 1.0]

		# Postures of human
		self.head_poses = [[1.0, 1.0, 1.0] for x in range(200)]
		self.body_poses = [[-100.0, -100.0, -100.0] for x in range(200)]
		self.eye_poses = [[-100.0, -100.0, -100.0] for x in range(200)]
		self.left_eye_landmark = 1
		self.right_eye_landmark = 1
		self.left_eye_gaze = 1
		self.right_eye_gaze = 1
		self.calib_center_eyes = [1.0, 1.0, 1.0]
		self.human_state = 'Standard' # Action recognition result

		# Stabilizers
		self.stabilizers = StabilizersWithKalmanFilter()

		# Estimation flag
		self.face_detection_flag = False
		self.head_pose_estimation_flag = False
		self.body_pose_estimation_flag = False
		self.gaze_estimation_flag = False

	# When the tracker did not estimate the value from human, the last value tracked in last frame is appended into the list.
	def _put_dummy_data(self, *lists):
		for list in lists:
			list.pop(0)
			list.append(list[-1])

	def _put_data(self, datas, type='center_eyes'):
		# Specify the list and stabilizer
		if type == 'center_eyes':
			stabilizer = self.stabilizers.center_eye_stabilizers
			save_list = self.center_eyes
			self.face_detection_flag = True
		elif type == 'center_mouths':
			stabilizer = self.stabilizers.center_mouth_stabilizers
			save_list = self.center_mouths
		elif type == 'left_shoulders':
			stabilizer = self.stabilizers.left_shoulder_stabilizers
			save_list = self.left_shoulders
		elif type == 'right_shoulders':
			stabilizer = self.stabilizers.right_shoulder_stabilizers
			save_list = self.right_shoulders
		elif type == 'center_stomachs':
			stabilizer = self.stabilizers.center_stomach_stabilizers
			save_list = self.center_stomachs
		elif type == 'head_poses':
			stabilizer = self.stabilizers.head_pose_stabilizers
			save_list = self.head_poses
			self.head_pose_estimation_flag = True
		elif type == 'body_poses':
			stabilizer = self.stabilizers.body_pose_stabilizers
			save_list = self.body_poses
			self.body_pose_estimation_flag = True
		elif type == 'eye_poses':
			stabilizer = self.stabilizers.eye_pose_stabilizers
			save_list = self.eye_poses
			self.gaze_estimation_flag = True


		# Operate to put the data to lists
		temp_list = datas
		#for index, data in enumerate(datas):
		#	stabilizer[index].update([data])
		#	temp_list.append(stabilizer[index].state[0][0])
		temp_list = np.array(temp_list)
		save_list.pop(0)
		save_list.append(temp_list)


# coding=utf8
import random

import numpy as np
import tensorflow as tf
from sklearn import svm

right0 = 0.0  # 记录预测为1且实际为1的结果数
error0 = 0  # 记录预测为1但实际为0的结果数
right1 = 0.0  # 记录预测为0且实际为0的结果数
error1 = 0  # 记录预测为0但实际为1的结果数

def dataprep():
	#
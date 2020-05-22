from tensorflow.python import pywrap_tensorflow
import os

checkpoint_path=os.path.join('D:/TensorflowTest/taitanic/model/model.ckpt-9001')
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    a=reader.get_tensor(key)
    print( 'tensor_name: ',key)
    print("a.shape:%s"%[a.shape])
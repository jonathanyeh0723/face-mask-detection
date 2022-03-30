# face-mask-detection

### My Process

```
from tensorflow.keras.preprocessing import image as I

test_img_path = "face_mask_test/nomask_4.jpg"

img = I.load_img(test_img_path, target_size=(150, 150))
print("1. Original image type: {}, shape: {}".format(type(img), img.size))

arr_img = I.img_to_array(img)
print("2. Preprocess image to array, type: {}, shape: {}".format(type(arr_img), arr_img.shape))

dim_added_img = np.expand_dims(arr_img, axis=0)
print("3. Using numpy to add image dimension to BHWC, type: {}, shape: {}".format(type(dim_added_img), dim_added_img.shape))
```

Output
1. Original image type: <class 'PIL.Image.Image'>, shape: (150, 150)
2. Preprocess image to array, type: <class 'numpy.ndarray'>, shape: (150, 150, 3)
3. Using numpy to add image dimension to BHWC, type: <class 'numpy.ndarray'>, shape: (1, 150, 150, 3)

Note:

PIL.Image.Image ---> img.size

numpy.ndarray ---> img.shape

### My Inference

```
from openvino.runtime import Core

core = Core()
model = core.read_model('C:/Users/Jonathan/saved_model.xml')

compiled_model = core.compile_model(model, 'CPU')
results = compiled_model.infer_new_request({0: images})

print(results)
print(type(results))
print(results.values())

inference = list(results.values())[0]
print(inference)
```

{<ConstOutput: names[Func/StatefulPartitionedCall/output/_11:0, Identity:0, StatefulPartitionedCall/Identity:0, StatefulPartitionedCall/sequential/dense_2/Sigmoid:0] shape{1,1} type: f32>: array([[0.]], dtype=float32)}

<class 'dict'>

dict_values([array([[0.]], dtype=float32)])

[[0.]]

The repo is forked for EE451 Project, aiming to apply MobileNetV2 Network with Cpp CUDA and optimize its performance by parallelism. The origin readme.md is localed at pythonVersion/README.md

## Bug to fixed
Attention, TensorFlow v2.16.1 cannot find all the cuda lib. Try to downgrade or try

```python
TF_CPP_MAX_VLOG_LEVEL=3 python -c "import tensorflow as tf;print(tf.config.list_physical_devices('GPU'))"
```
Some lib coule be no found and you should add them manually.




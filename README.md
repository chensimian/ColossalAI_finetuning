# ColossalAI微调LLaMA2

## 使用

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

```bash
运行 colossalai check -i 查看环境是否安装成功
------------ Environment ------------
Colossal-AI version: 0.3.4
PyTorch version: 2.0.0
System CUDA version: 11.3
CUDA version required by PyTorch: 11.7
```

```bash
运行其他加速选项flsh attention 与 xformers, 需要自行安装
注意安装apex选择的版本为23.05, 运行以下命令进行安装
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cuda_ext" ./
```
### 2. 性能测试脚本

运行scripts文件夹里面的bash文件即可，有训练平均吞吐以及GPU的 TFLOPS等性能指标


### 3. 微调llama2模型

运行 run.sh 即可，注意数据集的格式需与dataset里面的数据集格式一致，其中task_name为数据集名称，运行输出文件保存在指定目录中，同样有性能评测结果输出。


# 我的代码实现

## 数据预处理

首先需要将udacity的数据集里的图片名字重命名一下

```python
# rename.py
import pandas as pd
import numpy as np
import os.path

if __name__ == "__main__":
    img_folder = 'E:\\Lab\\research\\MUNIT\\inputs\\test_2414\\test\\center'
    num = 1
    for img_name in os.listdir(img_folder):
        old_name = os.path.join(img_folder, img_name)
        new_name = os.path.join(img_folder, "sunny_" + str(num) + '.jpg')
        os.rename(old_name, new_name)
        num += 1
```

接着重写一份csv，记录图片名和对应的正确的转向角

```python
import pandas as pd
import numpy as np


if __name__ == "__main__":
    data = pd.read_csv("E:\\Lab\\research\\MUNIT\\inputs\\test_2414\\test\\final_example.csv")
    data_img = []
    for i in range(1, 5615):
        data_img.append("sunny_" + str(i) + ".jpg")
    data_steering_angle = data["steering_angle"].tolist()
    target = pd.DataFrame({"image": data_img, "steering_angle": data_steering_angle})
    target.to_csv("E:\\Lab\\research\\MUNIT\\inputs\\test_2414\\test\\sunny.csv", index=False)

```

## 编写MUNIT脚本

MUNIT采用终端传参方式启动，所以需要编写对应的自动化脚本，我的电脑是win11，所以使用批处理指令。

```python
import os

if __name__ == "__main__":
    f = open("scripts/sunny2snow_night.cmd", "a")
    order = "python ../test.py --config ../configs/snow_night.yaml --output_folder ../outputs/snow_night --checkpoint ../models/snow_night.pt --a2b 1 --num_style 1"
    for img_name in os.listdir("E:\\Lab\\research\\MUNIT\\inputs\\sunny"):
        f.write(order + " --input E:\\Lab\\research\\MUNIT\\inputs\\sunny\\" + img_name + "\n")
    f.close()
```

## 修改test.py

为了方便第二阶段的自动驾驶模型输入的处理，生成图片的命名格式为：天气 + 序号

将下图的部分修改，即可得到我们需要的图片命名方式。

![image-20230413125240009](http://kiyotakawang.oss-cn-hangzhou.aliyuncs.com/img/image-20230413125240009.png)
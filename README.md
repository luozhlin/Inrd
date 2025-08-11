## 1. 数据预处理

### 1.1 原始数据

原始数据位于 data 文件夹下 raw_data 文件夹中
* KSC: 512, 614, 176
	* KSC.mat 
	* https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Kennedy_Space_Center_(KSC)
* WDC: 1280, 307, 191
	* wdc.tif
	* https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html
* WHU_Hi_LongKou: 550, 400, 270
	* WHU_Hi_LongKou.mat
	* https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
### 1.2 处理后数据
将所有数据裁剪成 256, 256 的大小，以水体为主，处理后数据在 data 文件夹下
#### 1.2.1 裁剪区域
* KSC: \[200, 200+255\], \[1, 1+255\]
* WDC: \[129, 129+255\], \[180, 180+255\]
* WHU_Hi_LongKou: \[50, 305\], \[50, 305\]
#### 1.2.2 denoise
位于 denoise 文件夹中
* 包含 gt, input, sigma，分别对应 ground truth, observation 和 噪声标准差
* 通过对gt 加标准差为 sigma 的噪声得到 input，都已经归一化

裁剪和加噪通过 data 文件夹下 crop_denoise_demo.m 实现
#### 1.2.3 super resolution
 位于 super_resolution 文件夹中，由 super_resolution.py实现
* 包含 gt, input, scale, sigma
* scale=0.5, 0.25, 0.125 对应 2x 4x 8x
* 对 gt 进行高斯模糊后下采样，再进行加噪得到 input
* 需要用到 guided_diffusion 中的 core.py，与 HIRDiff 保持一致
#### 1.2.4 inpainting
位于 inpainting 文件夹中，由 inpainting.py 实现
* 包含 gt，input，mask，sigma
* mask 表示掩码位置，由 0,1组成，mask=0 表示该位置缺失，该位置元素设为 0
	* KSC 数据集例外，mask=0 的位置元素设为 1，因为 KSC 自身数据接近 0，设为 0 并不会破坏原图像
* 对 gt 的 mask 位置设为 0 后，其余可观测的位置再添加噪声，得到 input

最后得到的数据范围因为加噪会超出$[0,1]$

## 2. 方法(INR-Diff)

Suppose we have $R=4,5,6,\dots$ splits, each with $3$ channel.
1. Initialize $E$ with $R$ bands and $INR$.
	1. main_re.py中RRQR 选择 band![image.png](https://raw.githubusercontent.com/luozhlin/Image/main/Obsidian/202508071242320.png)
	2. rsfac_grad_gaussian_diffusion_split.py 中计算 $E$ 和初始化 $INR$![image.png](https://raw.githubusercontent.com/luozhlin/Image/main/Obsidian/202508071245773.png)

2. Spliting method: $S=\{[1,2,3], [2,3,4],\dots,[R-2,R-1,1], [R-1,1,2]\},\quad |S|=R$.
	1. rsfac_grad_gaussian_diffusion_split.py中拆分索引![image.png](https://raw.githubusercontent.com/luozhlin/Image/main/Obsidian/202508071250034.png)

3. Start with noise $x_{T}$, shape $(H,W,R)$
4. For $t=T,T-1,\dots,0$
	1. Estimate $\hat{x}_{0}^i,\quad i\in S$. 
		1. 根据拆分索引计算每个拆分部分估计的$\hat{x}_{0}^i.$ ![](https://raw.githubusercontent.com/luozhlin/Image/main/Obsidian/202508071251170.png)model_output 对应 diffsuion 估计的噪声；pred_xstart 对应未归一化的$\hat{x}_{0}^i$ ;  xhat 则是归一化后的，后续恢复到原始图像也是基于 xhat
		2. 将每个划分的结果储存在 list 里面
	2. Obtain $\hat{x}_{0}$ by combining $\hat{x}_{0}^i$.
		1. Suppose $r=1,\dots,R$ is the band index
		2. $\hat{x}_{0}[r] =\text{mean}(\hat{x}_{0}^r[1]+\hat{x}_{0}^{r-1}[2]+\hat{x}_{0}^{r-2}[3])$.
		3. 对应通道上进行平均合并![image.png](https://raw.githubusercontent.com/luozhlin/Image/main/Obsidian/202508071255776.png)

	3. If $t\leq t_{0}$:
		1. Update $E$ by INR with loss function $L(\hat{x}_{0},E)$.![image.png](https://raw.githubusercontent.com/luozhlin/Image/main/Obsidian/202508071300146.png)

		2. Sample $x_{t-1}$ by DDIM + $\nabla L$
	
	4. else: 
		1. Sample $x_{t-1}$ by DDIM + $\nabla L$![image.png](https://raw.githubusercontent.com/luozhlin/Image/main/Obsidian/202508071301495.png)

### 2.1 HIRDiff
* 以下命令是运行 **HIRdiff** 使用，main.py
**denoise**
```
python main.py -eta1 16 -eta2 10 --k 8 -step 20 -dn Houston --task denoise --task_params 50 -gpu 0 --beta_schedule exp
```

**super resolution**

```
python main.py -eta1 500 -eta2 12 --k 8 -step 20 -dn WDC --task sr --task_params 0.25 -gpu 0 --beta_schedule exp
```
* 对于 2x 超分，需要调整eta1 的参数为 100 150
* KSC数据集除外，不需要调整，调整结果好像更好，但是保存的数据是未调整的

**inpainting**

```
python main.py -eta1 8 -eta2 6 --k 5 -step 20 -dn Salinas --task inpainting --task_params 0.8 -gpu 0 --beta_schedule exp
```

### 2.2 INRDiff

如果运行 **INRdiff**，只需要将 main.py 改成 main_inr.py，关联的函数在create_inr.py 和rsfac_grad_gaussian_diffusion_split.py 中
```
python main_inr.py -eta1 16 -eta2 10 --k 8 -step 20 -dn Houston --task denoise --task_params 50 -gpu 0 --beta_schedule exp

python main_inr.py -eta1 500 -eta2 12 --k 8 -step 20 -dn WDC --task sr --task_params 0.25 -gpu 0 --beta_schedule exp 

python main_inr.py -eta1 500 -eta2 12 --k 8 -step 20 -dn WDC --task sr --task_params 0.25 -gpu 0 --beta_schedule exp
```

2x超分可修改eta1 的参数，一般设为 80,100 左右
## 3. 对比方法
其他对比方法位于 other_methods 文件夹中，HIRdiff 除外，结果都储存在 results 中
### 3.1 denoise

#### 3.1.1 BM4D
* 直接 pip install bm4d，其他配置环境与 HIRDiff 一样
* 位于 denoise_BM4D 文件夹中
#### 3.1.2 NGMeet
* https://github.com/quanmingyao/NGMeet/
* 位于 denoise_NGMeet文件夹中，结果储存在 results 中
* 执行文件为 NGMeet_denoise.m
![image.png](https://raw.githubusercontent.com/luozhlin/Image/main/Obsidian/202508071430575.png)
* 如果要调整，参数设置位于 ParSetH.m 里面
#### 3.1.3 ETPTV
* https://github.com/chuchulyf/ETPTV
* 位于 denoise_ETPTV 文件夹中
* 执行文件为ETPTV_denoise.m

#### 3.1.4 TDSAT
* https://github.com/Featherrain/TDSAT
* 运行 hsi_test.py
```
python hsi_test.py -a tdsat -r -rp ./pretrained/model_epoch_50.pth --gpu-ids 0
```

### 3.2 Super resolution

### 3.3 Inpainting

#### 3.3.1 TRLRF
* https://github.com/yuanlonghao/TRLRF
* 位于 inpainting_TRLRF 文件夹中Exp_quick_test.m
#### 3.3.2 FCTN
* https://github.com/YuBangZheng/code_FCTN
* inpainting_FCTN中运行 Demo_HSV.m

#### 3.3.3 LRTC-TNN
* https://github.com/canyilu/Tensor-robust-PCA-and-tensor-completion-under-linear-transform
* 运行demo_ltrc_tnn_transform.m

## 4. 结果

### Denoising

#### KSC

| sigma   | 30             | 50             | 70             |
| ------- | -------------- | -------------- | -------------- |
| BM4D    | 33.34/0.95     | 29.73/0.88     | 27.93/0.82     |
| NGMeet  | 41.05/0.94     | 39.61/0.93     | 39.12/0.93     |
| ETPTV   | 39.36/0.94     | 38.22/0.93     | 36.75/0.91     |
| DIP2d   |                |                |                |
| TDSAT   | 38.83/0.94     |  37.20/0.90    | 38.36/0.92     |
| HIRDiff | 41.55/0.88     | 40.52/0.87     | 38.99/0.83     |
| Ours    | **42.87/0.91** | **42.71/0.91** | **42.78/0.85** |

#### WDC

| sigma   | 30             | 50             | 70             |
| ------- | -------------- | -------------- | -------------- |
| BM4D    | 35.46/0.93     | 32.7/0.88      | 31.07/0.83     |
| NGMeet  | **45.07/0.98** | **42.27/0.97** | **40.65/0.95** |
| ETPTV   | 42.06/0.97     | 37.49/0.93     | 34.14/0.86     |
| DIP2d   |                |                |                |
| TDSAT   | 36.85/0.93     | 34.78/0.89     | 33.68/0.85     |
| HIRDiff | 40.06/0.95     | 39.21/0.94     | 37.97/0.92     |
| Ours    | 42.67/0.97     | 41.89/0.96     | 40.21/0.94     |

#### WHU_LongKou

| sigma   | 30             | 50             | 70             |
| ------- | -------------- | -------------- | -------------- |
| BM4D    | 35.97/0.92     | 33.78/0.87     | 32.30/0.82     |
| NGMeet  | **42.84/0.98** | **40.05/0.96** | **38.74/0.95** |
| ETPTV   | 40.52/0.96     | 38.27/0.94     | 36.81/0.92     |
| DIP2d   |                |                |                |
| TDSAT   | 35.73/0.90     | 34.22/0.87     | 33.04/0.85     |
| HIRDiff | 38.98/0.94     | 38.56/0.94     | 37.55/0.93     |
| Ours    | *39.51/0.95*   | *39.04/0.95*   | *38.25/0.94*   |

### Super resolution

#### KSC

| scale   | 2x         | 4x         | 8x         |
| ------- | ---------- | ---------- | ---------- |
|         |            |            |            |
|         |            |            |            |
| HIRDiff | 38.88/0.90 | 36.02/0.74 | 36.45/0.61 |
| Ours    | 40.76/0.86 | 42.14/0.89 | 41.89/0.91 |
#### WDC


| scale   | 2x         | 4x         | 8x         |
| ------- | ---------- | ---------- | ---------- |
|         |            |            |            |
|         |            |            |            |
|         |            |            |            |
| HIRDiff | 37.18/0.86 | 35.29/0.83 | 33.03/0.78 |
| Ours    | 36.01/0.85 | 34.51/0.78 | 34.22/0.76 |
#### WHU_LongKou

| scale   | 2x         | 4x         | 8x         |
| ------- | ---------- | ---------- | ---------- |
|         |            |            |            |
|         |            |            |            |
|         |            |            |            |
| HIRDiff | 34.50/0.87 | 33.62/0.85 | 30.43/0.78 |
| Ours    | 35.20/0.89 | 33.54/0.85 | 30.62/0.77 |

### Inpainting

#### KSC

| rate     | 0.5            | 0.6            | 0.7            |
| -------- | -------------- | -------------- | -------------- |
| TRLRF    | 35.59/0.83     | 34.39/0.81     | 24.60/0.40     |
| FCTN     | 42.04/0.90     | 41.35/0.89     | 40.42/0.86     |
| LRTC-TNN | 31.19/0.77     | 30.54/0.75     | 29.93/0.73     |
| HIRDiff  | 40.25/0.55     | 39.66/0.53     | 39.37/0.52     |
| Ours     | **42.63/0.91** | **42.64/0.91** | **42.53/0.87** |

#### WDC

| rate     | 0.5        | 0.6        | 0.7        |
| -------- | ---------- | ---------- | ---------- |
| TRLRF    | 38.31/0.88 | 34.37/0.80 | 32.00/0.72 |
| FCTN     | **39.72/0.89** | **38.59/0.87** | **37.31/0.82** |
| LRTC-TNN | 37.80/0.90 | 36.25/0.87 | 34.83/0.82 |
| HIRDiff  | 37.63/0.89 | 37.45/0.88 | 37.22/0.88 |
| Ours     | 39.02/0.91 | 38.12/0.90 | 37.26/0.87 |

#### WHU_LongKou

| rate     | 0.5        | 0.6        | 0.7        |
| -------- | ---------- | ---------- | ---------- |
| TRLRF    | 35.61/0.90 | 33.17/0.83 | 28.20/0.62 |
| FCTN     | 36.25/0.92 | 34.85/0.90 | 32.66/0.86 |
| LRTC-TNN | 36.41/0.92 | 34.72/0.90 | 33.09/0.87 |
| HIRDiff  | 35.73/0.90 | 35.20/0.90 | 34.77/0.90 |
| Ours     | **36.41/0.92** | **35.85/0.91** | **34.97/0.90** |
|          |            |            |            |

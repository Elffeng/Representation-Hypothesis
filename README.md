# Representation-Hypothesis
python = 3.11
pip install -r requirements.txt
1. 从模型中提取特征
   文本模型
python extract_features.py --dataset minhuh/prh --subset wit_1024 --modelset val --modality language --pool avg
   视觉模型
python extract_features.py --dataset minhuh/prh --subset wit_1024 --modelset val --modality vision --pool cls
2. 找视觉模型的对齐
   python measure_alignment.py --dataset minhuh/prh --subset wit_1024 --modelset val \
        --modality_x language --pool_x avg --modality_y vision --pool_y cls
3. 在examples的文件夹中上传了第二第三幅图的画图代码，通过更改模型和数据集便可画出

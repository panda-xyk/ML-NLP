# Chinese NER [原项目地址](https://github.com/yanwii/ChineseNER)

**参考--[基于BERT预训练的中文命名实体识别TensorFlow实现](https://blog.csdn.net/macanv/article/details/85684284)**

基于Bi-GRU + CRF 的中文机构名、人名识别
集成GOOGLE BERT模型

## 下载bert模型

```bash
  wget -c https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
```

放到根目录 **bert_model** 下

## 目录结构如下：

```text
|----ChineseNER # 项目根目录
  |----ber_model
    |----chinese_L-12_H-768_A-12 # bert预训练模型
      |----bert_config.json
      |----bert_model.ckpt.data-00000-of-00001 # 存储网络参数的数值
      |----bert_model.ckpt.index # 存储每一层的名字
      |----bert_model.ckpt.meta # 存储图的结构
      |----vocab.txt # 词汇表，以汉字为单位
  |----data
  |----model # 训练好的ner模型，执行训练步骤后会自动生成,如果是重新训练的话，需要把此目录删除，否则会因为已经存在模型而报错
  |----.gitignore # 忽略文件
  |----bert_data_utils.py # 调用bert模型时的一些数据处理方法
  |----data_utils.py # 普通的数据预处理相关的一些方法（不使用bert）
  |----helper.py # argparse相关
  |----model.py # 入口函数
  |----README.md # readme文件
  |----utils.py # 通用工具函数
```

## 用法

**每次重新训练的话，需要把model目录删除，否则会因为已经存在模型而报错**

```bash
  # 训练
  # 使用bert模型
  python3 model.py -e train -m bert
  # 使用一般模型
  python3 model.py -e train

  # 预测
  # 使用bert模型
  python3 model.py -e predict -m bert
  # 使用一般模型
  python3 model.py -e predict
```

## 介绍

### bert 模型的加载和使用

```python
  def bert_layer(self):
      # 加载bert配置文件
      bert_config = modeling.BertConfig.from_json_file(ARGS.bert_config)

      # 创建bert模型　
      model = modeling.BertModel(
          config=bert_config,
          is_training=self.is_training,
          input_ids=self.input_ids,
          input_mask=self.input_mask,
          token_type_ids=self.segment_ids,
          use_one_hot_embeddings=False
      )
      # 加载词向量
      self.embedded = model.get_sequence_output()
      self.model_inputs = tf.nn.dropout(
          self.embedded, self.dropout
      )
```

### bert 优化器

```python
  self.train_op = create_optimizer(
      self.loss, self.learning_rate, num_train_steps, num_warmup_steps, False
  )
```

## 例子

```python
  > 金树良先生，董事，硕士。现任北方国际信托股份有限公司总经济师。曾任职于北京大学经济学院国际经济系。1992年7月起历任海南省证券公司副总裁、北京华宇世纪投资有限公司副总裁、昆仑证券有限责任公司总裁、北方国际信托股份有限公司资产管理部总经理及公司总经理助理兼资产管理部总经理、渤海财产保险股份有限公司常务副总经理及总经理、北方国际信托股份有限公司总经理助理。
  >   [
          {
            "begin": 14,
            "end": 26,
            "entity": "北方国际信托股份有限公司",
            "type": "ORG"
          },
          {
            "begin": 70,
            "end": 82,
            "entity": "北京华宇世纪投资有限公司",
            "type": "ORG"
          },
          {
            "begin": 99,
            "end": 111,
            "entity": "北方国际信托股份有限公司",
            "type": "ORG"
          },
          {
            "begin": 160,
            "end": 172,
            "entity": "北方国际信托股份有限公司",
            "type": "ORG"
          },
          {
            "begin": 0,
            "end": 3,
            "entity": "金树良",
            "type": "PER"
          }
      ]
```

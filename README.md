# PoseContrast


## Data configuration
- Human36M등의 각 데이터셋의 심볼릭링크는 각 convention에 맞게 설정하시면 될듯 합니다.  


- 이 두개는 아래 폴더를 다운받아서 대체하면 됩니다.  
```
./data/base_data  
./experiment  
```

[[Link](https://drive.google.com/drive/folders/1gFC5LML7oD48w542rFNpiAzUOtMOM7Nc?usp=sharing)]



## Experiment setting
- asset/yaml/example.yml 란에 각 option에 대한 설명을 붙혀두었습니다.


Contrastive learning

python main/train.py --gpu 0,1,2,3 --cfg ./asset/yaml/train_contrastive.yml


HPS (human body pose&shape) learning

python main/train.py --gpu 0,1 --cfg ./asset/yaml/train_body.yml


HPS (human body pose&shape) evaluation

python main/test.py --gpu 0 --cfg ./asset/yaml/eval_body.yml

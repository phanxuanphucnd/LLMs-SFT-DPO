## Table of Contents

- [Structure Folders](#structure)
- [Training](#training)
- [Inference](#inference)



### Structure Folders

- `data`: Folder chứa data
    - `data/testset`: Folder chứa data test
- `models`: Folder lưu trữ model sau khi fine-tuning (adapter)
- `src`: Folder chứa code
- `storages`: Folder lưu trữ model sử dụng để inference


### Training

```js
bash sft_run.sh
```


### Inference

```js
python inference.py
```

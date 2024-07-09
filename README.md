```sh
conda create -n opd python=3.9
conda activate opd
pip install -r requirements.txt
```


## Dataset
Download  [OPDMulti](https://docs.google.com/forms/d/e/1FAIpQLSeG1Jafcy9P_OFBJ8WffYt6WJsJszXPqKIgQz0tGTYYuhm4SA/viewform?vc=0&c=0&w=1&flr=0) dataset (7.2G) and extract it inside `./dataset/` folder. Make sure the data is in [this](https://github.com/3dlg-hcvc/OPDMulti/blob/master/data/README.md#downloaded-data-organization) format.  You can follow [these](https://github.com/3dlg-hcvc/OPDMulti/blob/master/data/README.md#data-processing-procedure) steps if you want to convert your data to OPDMulti dataset. To try our model on OPDSynth and OPDReal datasets, download the data from [OPD](https://github.com/3dlg-hcvc/OPD#dataset) repository.


## Training
To train from the scratch, you can use the below commands. The output will include evaluation results on the val set.

```sh
cd opdformer
python train.py \
--config-file <MODEL_CONFIG> \
--output-dir <OUTPUT_DIR> \
--data-path <PATH_TO_DATASET> \
--input-format <RGB/depth/RGBD> \
--model_attr_path <PATH_TO_ATTR> 
```
* `<MODEL_CONFIG>`: the config file path for different model variants can be found in the table [OPDMulti](#opdmulti) "Model Name" column.
    
* Dataset:
    * --data-path `OPDMulti/MotionDataset_h5`
    * --model_attr_path: ` OPDMulti/obj_info.json `
* You can add the following command to use the model weights, pretrained on OPDReal dataset. We finetune this model on OPDMulti dataset:

  `--opts MODEL.WEIGHTS <PPRETRAINED_MODEL>`

## Evaluation
To evaluate, use the following command:

```sh
python evaluate_on_log.py \
--config-file <MODEL_CONFIG> \
--output-dir <OUTPUT_DIR> \
--data-path <PATH_TO_DATASET> \
--input-format <RGB/depth/RGBD> \
--model_attr_path <PATH_TO_ATTR> \
--opts MODEL.WEIGHTS <PPRETRAINED_MODEL>
```

* Evaluate on test set: `--opts MODEL.WEIGHTS <PPRETRAINED_MODEL> DATASETS.TEST "('MotionNet_test',)"`.
* To evaluate directly on pre-saved inference file, pass the file path as an argument `--inference-file <PATH_TO_INFERENCE_FILE>`.

## Pretrained-Models



## Visualization
The visualization code is based on [OPD](https://github.com/3dlg-hcvc/OPD.git) repository. We only support visualization based on raw dataset format ([download link](https://docs.google.com/forms/d/e/1FAIpQLSeG1Jafcy9P_OFBJ8WffYt6WJsJszXPqKIgQz0tGTYYuhm4SA/viewform?vc=0&c=0&w=1&flr=0) (5.0G)).

And the visualization uses the inference file, which can be obtained after the evaluation.
* Visualize the GT with 1000 random images in val set 
  ```sh
  cd opdformer
  python render_gt.py \
  --output-dir vis_output \
  --data-path <PATH_TO_DATASET> \
  --valid-image <IMAGE_LIST_FILE> \
  --is-real
  ```
* Visualize the PREDICTION with 1000 random images in val set
  ```sh
  cd opdformer
  python render_pred.py \
  --output-dir vis_output \
  --data-path <PATH_TO_DATASET> \
  --model_attr_path <PATH_TO_ATTR> \
  --valid-image <IMAGE_LIST_FILE> \
  --inference-file <PATH_TO_INFERENCE_FILE> \
  --score-threshold 0.8 \
  --update-all \
  --is-real
  ```
  * --data-path `dataset/MotionDataset`
  * --valid_image `dataset/MotionDataset/valid_1000.json`

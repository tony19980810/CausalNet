 # 🚀 Rethinking Key-frame-based Micro-expression Recognition: A Robust and Accurate Framework Against Key-frame Errors

[![license](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/tony19980810/CausalNet/blob/main/LICENSE)

[![arXiv](https://img.shields.io/badge/arXiv-2508.06640-red)](https://arxiv.org/abs/2508.06640)


[ICCV 25] Official repository for the paper "Rethinking Key-frame-based Micro-expression Recognition: A Robust and Accurate Framework Against Key-frame Errors" 



## Abstract

Micro-expression recognition (MER) is a highly challenging task in affective computing. With the reduced-sized micro-expression (ME) input that contains key information based on key-frame indexes, key-frame-based methods have significantly improved the performance of MER. However, most of these methods focus on improving the performance with relatively accurate key-frame indexes, while ignoring the difficulty of obtaining accurate key-frame indexes and the objective existence of key-frame index errors, which impedes them from moving towards practical applications. In this paper, we propose CausalNet, a novel framework to achieve robust MER facing key-frame index errors while maintaining accurate recognition. To enhance robustness, CausalNet takes the representation of the entire ME sequence as the input. To address the information redundancy brought by the complete ME range input and maintain accurate recognition, first, the Causal Motion Position Learning Module (CMPLM) is proposed to help the model locate the muscle movement areas related to Action Units (AUs), thereby reducing the attention to other redundant areas. Second, the Causal Attention Block (CAB) is proposed to deeply learn the causal relationships between the muscle contraction and relaxation movements in MEs. Empirical experiments have demonstrated that on popular ME benchmarks, the CausalNet has achieved robust MER under different levels of key-frame index noise. Meanwhile, it has surpassed state‑of‑the‑art (SOTA) methods on several standard MER benchmarks when using the provided annotated key‑frames.


![image](image/Main.png)

## 📄Paper

Main Paper: [link](https://arxiv.org/abs/2508.06640)

## ✨Video

10-minute quick understanding: [link](https://www.youtube.com/watch?v=MVudRJXM8iE&t=1s)

## 🛠️Setup


Install the environment:

```bash
pip install -r requirements.txt
```

## 🏋️‍♂Training

Please use the following code for training to find the optimal parameters:


```bash
python main_train_for_parameter_tuning.py
```

The prediction results and ground truth will be saved in text files, accompanied by a composite dataset result in XLSX format. All these files will be stored in a folder named after the parameter values within the ./results directory. Additionally, the overall results will be recorded in result_summary.txt for easy viewing.

---

## 🧩Evaluation

After training, the results will be saved in the txt files under the results directory. Please place all txt files in the ./results directory and use the following code to calculate the metrics:

```bash
python calculate_all_results.py
python calculate_all_results_CASMEII.py
python calculate_all_results_SAMM.py
python calculate_all_results_SMIC.py
```
We also provide one case of our trained weights [link](https://pan.baidu.com/s/1GHcStkaPE7iEh1_oUs2Wmw?pwd=6666), and the results are as follows:

| Method     | Pub        | CASME II UF1 | CASME II UAR | CASME II ACC | SMIC UF1 | SMIC UAR | SMIC ACC | SAMM UF1 | SAMM UAR | SAMM ACC | Composite UF1 | Composite UAR | Composite ACC |
|:------------|:-----------|-------------:|-------------:|-------------:|----------:|----------:|----------:|----------:|----------:|----------:|--------------:|--------------:|--------------:|
| CausalNet   | Proposed   | **98.11**    | **98.58**    | **98.62**    | **83.90** | **84.52** | **84.15** | **87.80** |   **85.82**   | **91.73** |     **89.76**    |     **90.20**    |     **91.18**     |

Pleas download the weights, and use the following code for evaluation:
```bash
python eval.py
```




##  🙏Acknowledgements


The framework of the code are based on the excellent work of [HTNet](https://github.com/wangzhifengharrison/HTNet). The experiments are built upon the excellent work of [OffTANet](https://github.com/ECNU-Cross-Innovation-Lab/PRICAI2021-Off-TANet), [MMNet](https://github.com/muse1998/MMNet), [SRMCL](https://github.com/pakchoi-php/SRMCL). The dataset is composed based on the excellent work of [CASME II](http://casme.psych.ac.cn/casme/c2), [SMIC](https://ieeexplore.ieee.org/document/6553717), [SAMM](https://ieeexplore.ieee.org/document/7492264), and [MMEW](https://github.com/benxianyeteam/MMEW-Dataset). We would like to express our gratitude for their open-source efforts.

## 🔮Citation

If you find this repo useful for your research, please cite the paper

```

@misc{zhang2025rethinkingkeyframebasedmicroexpressionrecognition,
      title={Rethinking Key-frame-based Micro-expression Recognition: A Robust and Accurate Framework Against Key-frame Errors}, 
      author={Zheyuan Zhang and Weihao Tang and Hong Chen},
      year={2025},
      eprint={2508.06640},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.06640}, 
}

@InProceedings{Zhang_2025_ICCV,
    author    = {Zhang, Zheyuan and Tang, Weihao and Chen, Hong},
    title     = {Rethinking Key-frame-based Micro-expression Recognition: A Robust and Accurate Framework Against Key-frame Errors},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {12274-12283}
}

```


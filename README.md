<p align="center">

  <h2 align="center">AToM: Aligning Text-to-Motion Model at Event-Level with GPT-4Vision Reward </h2>
  <p align="center">
    <strong>Haonan Han</strong></a><sup>1*</sup>
    · 
    <strong>Xiangzuo Wu</strong></a><sup>1*</sup>
    · 
    <strong>Huan Liao</strong></a><sup>1*</sup>
    ·
    <strong>Zunnan Xu</strong></a><sup>1</sup>
    ·
    <strong>Zhongyuan Hu</strong></a><sup>1</sup>
    ·
    <br>
    <strong>Ronghui Li</strong></a><sup>1</sup>
    ·
    <strong>Yachao Zhang</strong></a><sup>2†</sup>
    ·
    <strong>Xiu Li</strong></a><sup>1†</sup>
    <br>
    <sup>1</sup>Shenzhen International Graduate School, Tsinghua University  &nbsp;&nbsp;&nbsp; <sup>2</sup>School of Informatics, Xiamen University &nbsp;&nbsp;&nbsp;
    <br>
    </br>
        <a href=""><img src="https://img.shields.io/badge/ArXiv-2405.18525-brightgreen"></a> &nbsp; &nbsp;  &nbsp;
<a href="https://atom-motion.github.io/"><img src="https://img.shields.io/badge/Demo-AToM-purple"></a>&nbsp; &nbsp;  &nbsp;
<a href=""><img src="https://img.shields.io/badge/Dataset-MotionPrefer-blue"></a>
    </br> 

![Example Image](assets/pipeline.png)
    </br>
   Recently, text-to-motion models have opened new possibilities for creating realistic human motion with greater efficiency and flexibility. However, aligning motion generation with event-level textual descriptions presents unique challenges due to the complex relationship between textual prompts and desired motion outcomes. To address this, we introduce AToM, a framework that enhances the alignment between generated motion and text prompts by leveraging reward from GPT-4Vision. AToM comprises three main stages: Firstly, we construct a dataset MotionPrefer that pairs three types of event-level textual prompts with generated motions, which cover the integrity, temporal relationship and frequency of motion. Secondly, we design a paradigm that utilizes GPT-4Vision for detailed motion annotation, including visual data formatting, task-specific instructions and scoring rules for each sub-task. Finally, we fine-tune an existing text-to-motion model using reinforcement learning guided by this paradigm. Experimental results demonstrate that AToM significantly improves the event-level alignment quality of text-to-motion generation.
  </p>
    </p>
<!-- <div align="center"> -->

## Todo List
- [x] 🪄 Release on arxiv!
- [x] 🪄 Release demo page!
- [ ] 🔥 Release fine-grained motion preference dataset MotionPrefer including 80K preference pairs.
- [ ] 🔥 Release the instructions for prompt construction and alignment score annotations for different sub-tasks.
- [ ] 🔥 Release the fine-tuned model checkpoints based on preference data of different types.
- [ ] 🔥 Release more qualitative experimental results.



## Acknowledgement
We sincerely acknowledge and appreciate the exceptional open-source contributions that form the foundation of our work: [MotionGPT](https://github.com/OpenMotionLab/MotionGPT), [InstructMotion](https://github.com/THU-LYJ-Lab/InstructMotion).
## Citation

```


```

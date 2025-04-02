
<div align="center">
<h2>InstructRestore: Region-Customized Image Restoration with Human Instructions</h2>

<a href='http://arxiv.org/abs/2503.24357'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>


Shuaizheng Liu<sup>1,2</sup>
| Jianqi Ma<sup>1</sup> | 
Lingchen Sun<sup>1,2</sup> | 
Xiangtao Kong<sup>1,2</sup> | 
Lei Zhang<sup>1,2</sup>

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute
</div>

##  💡  Overview

![InstructRestore](figs/teasers1.png)

Our proposed **InstructionRestore** framework enables region-customized restoration following human instruction. 

(a) current methods tend to incorrectly restore the bokeh blur, while our method allows for adjustable control over the degree of blur based on user instructions. 

(b) existing methods fail to achieve region-specific enhancement intensities, while our approach can simultaneously suppress the over-enhancement in areas of building and improve the visual quality in areas of leaves.


##  🎨 Application
### Demo on Real-world Localized Enhancement
<img src="figs/localized_enhancement.png" alt="InstructRestore" width="600">

By following the instruction, the details in flowers are enhanced gradually while the other regions keeping almost unchanged.
### Demo on Controllable Bokeh Effects 
<img src="figs/controllable_bokeh.png" alt="InstructRestore" width="600">

By following the instruction, 

(a) Restoration with controlled bokeh effect while restoring foreground. 

(b) Restoration with varying foreground enhancement levels while preserving background bokeh.


### Comparisons with Other DM-Based global restoration Methods
(a) For the localized enhancement
![InstructRestore](figs/local_compare.png)

(b) For the preservation of bokeh effects
![InstructRestore](figs/bokeh_compare.png)

##  🍭 Achitecture
![InstructRestore](figs/architecture.png)

## 🌱  Dataset Construction Pipeline
![InstructRestore](figs/Dataset_construction.png)





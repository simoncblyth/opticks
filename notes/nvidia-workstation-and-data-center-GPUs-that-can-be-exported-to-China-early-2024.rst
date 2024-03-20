Review NVIDIA GPUs that can be exported from USA to China : Early 2024
=========================================================================

Workstation GPUs : RTX 5880 
--------------------------------

NVIDIA Launches RTX 5880 ProViz Card: Compliant with Sanctions, Available Globally
by Anton Shilov on January 9, 2024 5:00 PM EST 

* https://www.anandtech.com/show/21221/nvidia-launches-rtx-5880-proviz-card-compliant-with-sanctions-available-globally

* https://www.nvidia.com/en-us/design-visualization/rtx-5880/
* https://www.nvidia.com/en-us/design-visualization/desktop-graphics/

::

    RTX 6000   (restricted?)
    RTX 5880   (cut down for China?) 
    RTX 5000
    RTX 4500 
    RTX 4000


* https://resources.nvidia.com/en-us-design-viz-stories-ep/proviz-rtx-5880?lx=CCKW39&contentType=data-sheet
* https://www.cgchannel.com/2023/08/nvidia-unveils-rtx-4000-4500-and-5000-workstation-gpus/


Thoughts on current optimal workstation GPU for Opticks in China
-------------------------------------------------------------------

Comparing "NVIDIA RTX 5000 Ada Generation” (32GB) 
with "NVIDIA RTX A5000 (Ampere Generation)”
from the prior generation shows 2x performance improvement in many metrics according to the below:

* https://www.leadtek.com/eng/news/product_news_detail/1717


Performance of "NVIDIA RTX 5000 Ada Generation” is surpassed by 
"NVIDIA RTX 6000 Ada Generation” by around ~40% according to: 
* https://www.topcpu.net/en/cpu-c/rtx-6000-ada-generation-vs-rtx-5000-ada-generation


But the 6000 exceeds the US export threshold, according to:

* https://wccftech.com/us-adds-more-nvidia-gpus-to-china-ban-list-rtx-6000-ada-rtx-a6000-l4/


So that means that the "NVIDIA RTX 5000 Ada Generation” is close to 
being the most performant currently possible for ray tracing 
and AI workloads. 

There is however a GPU in-between the 5000 and 6000:
the "NVIDIA RTX 5880 Ada Generation”
with performance that is targeted just below the export threshold.  

* https://www.nvidia.com/en-us/design-visualization/rtx-5880/
* https://www.techpowerup.com/gpu-specs/rtx-5880-ada-generation.c4191

* https://www.anandtech.com/show/21221/nvidia-launches-rtx-5880-proviz-card-compliant-with-sanctions-available-globally


The above page has a table comparing the below GPUs, a few rows from the full table::

   RTX 6000 Ada        RTX 5880 Ada           RTX 5000 Ada

   48GB                48GB                   32GB            VRAM
   18,176              14,080                 12,800          cores 
   568                 440                    400             tensor cores
   142                 110                    100             RT cores
   300W                285W                   250W            power
   6,999 USD            ?                     4,000 USD       ~price 
   5828                4432                   4176            TPP 


TPP is "Total Processing Power” with export threshold of 4800 


* https://www.tomshardware.com/pc-components/gpus/nvidia-launches-another-sanctions-compliant-gpu-for-china-rtx-5880-ada-debuts-with-14080-cuda-cores-48gb-gddr6


I would suggest to ask the vendor(s) about "RTX 5880 Ada”, but 
as that was only released in January it might not become available 
within the needed timeframe. 


Comparing RTX 4090 with RTX 6000 Ada:

* https://irendering.net/compare-rtx-4090-vs-rtx-6000-ada-vs-rtx-a6000-for-content-creation/





Data Center GPUs : L20, L2 
-----------------------------

* H20 also : but that doesnt have RT cores

* https://videocardz.com/newz/nvidia-to-launch-hgx-h20-l20-and-l2-gpus-for-china

* https://www.tomshardware.com/pc-components/gpus/new-nvidia-ai-gpus-designed-to-get-around-us-export-bans-come-to-china-h20-l20-and-l2-to-fill-void-left-by-restricted-models
* https://techhq.com/2024/01/china-is-resisting-the-downgraded-ai-chips-from-nvidia-whats-next/


* https://docs.nvidia.com/grid/gpus-supported-by-vgpu.html

* https://www.guru3d.com/story/nvidia-introduces-new-product-line-for-china-adhering-to-us-export-regulations/

* https://techovedas.com/nvidia-to-launch-new-data-center-gpu-for-china-in-response-to-us-restrictions/







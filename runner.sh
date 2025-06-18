#!/bin/bash

# For original Brats comparing the effect of threshold and dilation
# python analysis.py # Both None
# python analysis.py --dilation 0
# python analysis.py --threshold 0
# python analysis.py --dilation 0 --threshold 0

# For Panoptica comparing the effect of matcher and threshold
python analysis_panoptica.py --matcher bipartite --threshold 0.000001
python analysis_panoptica.py --matcher bipartite --threshold 0.5

python analysis_panoptica.py --matcher naive --threshold 0.000001
python analysis_panoptica.py --matcher naive --threshold 0.5

#todo: Need to fix error where the part is the second label and not the next immediate one
# # For PartPanoptica comparing the effect of matcher and threshold
# python analysis_panoptica.py --style part --matcher bipartite --threshold 0.000001
# python analysis_panoptica.py --style part --matcher bipartite --threshold 0.5

# python analysis_panoptica.py --style part --matcher naive --threshold 0.000001
# python analysis_panoptica.py --style part --matcher naive --threshold 0.5
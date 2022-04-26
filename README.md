
# Robust Prototypical Network
This work is a project for the special topic course, Trustworthy Machine Learning (CSCI 6968), at Rensselaer Polytechnic Institute.

This work is a further improvement on the work of Snell et al. (2017), "Prototypical Networks for Few-shot Learning," (https://papers.nips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html), utilizing the work of Zhu et al. (2020), "Robust Re-weighting Prototypical Networks for Few-Shot Classification, " (https://dl.acm.org/doi/abs/10.1145/3449301.3449325) to produce more robust prototypes. 

The work also takes advantage of the information obtained from contrasting two prototypes, one computed from the support set and the other computed from the query set, to construct a new loss function which helps train ProtoNet for better generalization. 

Experiments and their data are in the jupyter notebook file (few-shot.ipynb).

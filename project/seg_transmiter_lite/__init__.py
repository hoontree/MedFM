"""Seg-TransMiter-lite prototype.

Bidirectional adapter-based knowledge transfer between TinyUSFM and SAM
for 3-class breast ultrasound segmentation
(0 = background, 1 = benign, 2 = malignant).

Entry points
------------
* :mod:`project.seg_transmiter_lite.sam_teacher_cache`
* :mod:`project.seg_transmiter_lite.train_tiny_with_sam_prior`
* :mod:`project.seg_transmiter_lite.train_sam_us_adapter`
* :mod:`project.seg_transmiter_lite.eval`

Reusable building blocks live alongside the rest of the codebase:
* ``model.seg_transmiter.adapters``
* ``utils.seg_transmiter_losses``
"""

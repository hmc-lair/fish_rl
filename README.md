# Fish RL project

```
conda create -n pytorch
conda activate pytorch
conda install gymnasium
conda install pytorch
conda install matplotlib
pip install ipython
pip install gymnasium[all]
```

Notes:
Camera is limited to 30 fps (period of 33.33 s)
Determined empirically and on the vendor's website
http://www.webcamerausb.com/elp-2megapixels-usb-webcam-full-hd-1080p-30fps-no-distortion-lens-driverless-cmos-ov2710-usb20-camera-module-for-scanning-vending-machine-p-334.html

Transmitter board seems to be limited to 312.5 commands per second (period of 0.0032 s)
Determined empirically

Robot wheel speed:
https://learn.parallax.com/support/reference/cyberbot-library-reference
L, R in 1/64th wheel turn increments/second. + is forward, - is backward.
Limits: +/- 128
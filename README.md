This fork tries to fix the problems related to the learning.

dependecies:
* Keras
* Pillow
* Matplotlib

With `yolo_char.py` you can train/test the YOLO algorithm.
Simply comment/uncomment sections at the end of file:
* `Sample` : Take a look at the data used.
* `Train` : Train the YOLO network.
* `Sample pred` : Check YOLO predictions.

All datas was generated on the fly.
The wheights of the network was saved in `weights.h5`.

By default the file `weights.h5` contains the values for a loss of ~0.002.
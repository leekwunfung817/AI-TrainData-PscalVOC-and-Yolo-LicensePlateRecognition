cd %~dp0
darknet detector test train.config.data yolov4-tiny-custom.cfg backup/yolov4-tiny-custom_60000.weights -thresh 0.25 -save_labels -dont_show -ext_output data/17_1_i_h_01649433_.jpg
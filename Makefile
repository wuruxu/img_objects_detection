all:
	gcc -march=native -o tf tfmain.c yuv_rgb.c cairo_jpg/src/cairo_jpg.c -Icairo_jpg/src -I/home/tf/dev/out/include/libxml2 -I/home/tf/dev/out/include/cairo -Iinclude -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart -Llib  -ltensorflow_framework -ltensorflow -I/home/tf/dev/out/include -L/home/tf/dev/out/lib -ljpeg -lturbojpeg -lcairo -lv4l2 -lmicrohttpd -lcrypto -lsqlite3 -lxml2

yuv2jpg: yuv2jpg.c
	gcc -march=native -o yuv2jpg yuv2jpg.c yuv_rgb.c -I/home/tf/dev/out/include -L/home/tf/dev/out/lib -lturbojpeg
go:
	./tf frozen_inference_graph.pb
	#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:lib:$(LD_LIBRARY_PATH)
	#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:lib:/home/tf/dev/out/lib:$LD_LIBRARY_PATH
	#export CUDA_VISIBLE_DEVICES=""
	#unset CUDA_VISIBLE_DEVICES
